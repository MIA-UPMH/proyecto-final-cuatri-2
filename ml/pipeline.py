from __future__ import annotations

from dataclasses import asdict
import io
import json
import tarfile
import time
from datetime import datetime
import os
from pathlib import Path

from sqlalchemy import create_engine, text

from train import TrainingCancelled, execute_training


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./respiratory_lab.db",
)
ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_ROOT", "/workspace/artifacts"))
PROMOTION_METRIC = os.getenv("PROMOTION_METRIC", "macro_f1")
DATASET_ROOT = Path(
    os.getenv(
        "DATASET_ROOT",
        "/workspace/files/Enf. Respiratorias_3ro/Respiratorios/chest_xray",
    )
)
FEEDBACK_ROOT = Path(os.getenv("FEEDBACK_ROOT", "/workspace/artifacts/validated_samples"))

# SageMaker / AWS settings (read from environment)
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_REGION = os.getenv("AWS_REGION", "us-east-2")
S3_BUCKET = os.getenv("S3_BUCKET", "")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN", "")
SAGEMAKER_INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.c5.xlarge")

# Official AWS DLC for PyTorch 2.4.0 CPU
PYTORCH_SAGEMAKER_IMAGE = (
    "763104351884.dkr.ecr.us-east-2.amazonaws.com"
    "/pytorch-training:2.4.0-cpu-py311-ubuntu22.04-sagemaker"
)


def _row_to_dict(row, keys):
    """Convert a DB row (tuple or RowProxy) to a dict."""
    if hasattr(row, "_mapping"):
        return dict(row._mapping)
    return dict(zip(keys, row))


def promote_if_better(conn, run_id: int, candidate_metric: float, candidate_accuracy: float) -> None:
    """Promote candidate if it beats the current champion."""
    current = conn.execute(
        text(
            "SELECT id, macro_f1, test_accuracy FROM training_runs "
            "WHERE is_promoted = true ORDER BY id DESC LIMIT 1"
        )
    ).fetchone()

    if current is None:
        conn.execute(text("UPDATE training_runs SET is_promoted = true WHERE id = :id"), {"id": run_id})
        return

    cur = _row_to_dict(current, ["id", "macro_f1", "test_accuracy"])
    current_metric = cur.get(PROMOTION_METRIC) or 0.0
    current_acc = cur.get("test_accuracy") or 0.0

    if candidate_metric > current_metric and candidate_accuracy >= current_acc:
        conn.execute(text("UPDATE training_runs SET is_promoted = false WHERE id = :id"), {"id": cur["id"]})
        conn.execute(text("UPDATE training_runs SET is_promoted = true WHERE id = :id"), {"id": run_id})


def _cancellation_state(engine, run_id: int) -> tuple[bool, str]:
    with engine.connect() as conn:
        row = conn.execute(
            text("SELECT status, cancellation_reason FROM training_runs WHERE id = :id"),
            {"id": run_id},
        ).fetchone()

    if row is None:
        return True, "Run no encontrado durante la cancelacion."

    state = _row_to_dict(row, ["status", "cancellation_reason"])
    reason = state.get("cancellation_reason") or "Cancelacion solicitada por el usuario."
    return state.get("status") == "cancel_requested", reason


# ─────────────────────────────────────────────────────────────────────────────
# SageMaker helpers
# ─────────────────────────────────────────────────────────────────────────────

def _upload_training_code_to_s3(s3_client) -> str:
    """Package train_sagemaker.py and upload to S3. Return s3:// URI."""
    script_path = Path(__file__).parent / "train_sagemaker.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(script_path), arcname="train_sagemaker.py")
    buf.seek(0)

    s3_key = "code/train_sagemaker.tar.gz"
    s3_client.put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf.read())
    uri = f"s3://{S3_BUCKET}/{s3_key}"
    print(f"[pipeline] Uploaded training code to {uri}")
    return uri


def _run_sagemaker(run_id: int, cfg: dict, log_event) -> dict | None:
    """
    Submit a SageMaker training job, poll until done, and return metrics dict.
    Returns None on terminal failure (run already marked failed in DB).
    """
    try:
        import boto3
    except ImportError:
        log_event("phase", "error", "boto3 not installed — cannot run SageMaker job")
        return None

    try:
        sm = boto3.client(
            "sagemaker",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
    except Exception as exc:
        log_event("phase", "error", f"Could not create AWS clients: {exc}")
        return None

    # Upload training script
    try:
        code_uri = _upload_training_code_to_s3(s3)
    except Exception as exc:
        log_event("phase", "error", f"Failed to upload training code: {exc}")
        return None

    instance_type = cfg.get("instance_type") or SAGEMAKER_INSTANCE_TYPE
    job_name = f"corte3-cnn-run-{run_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    hyperparameters = {
        "sagemaker_program": "train_sagemaker.py",
        "sagemaker_submit_directory": code_uri,
        "epochs": str(cfg.get("epochs", 12)),
        "batch-size": str(cfg.get("batch_size", 16)),
        "learning-rate": str(cfg.get("learning_rate", 0.0007)),
        "image-size": str(cfg.get("image_size", 224)),
        "optimizer": cfg.get("optimizer", "adam"),
        "dropout": str(cfg.get("dropout", 0.25)),
        "weight-decay": str(cfg.get("weight_decay", 0.0)),
        "gradient-clip": str(cfg.get("gradient_clip", 0.0)),
    }

    training_params: dict = {
        "TrainingJobName": job_name,
        "RoleArn": SAGEMAKER_ROLE_ARN,
        "AlgorithmSpecification": {
            "TrainingImage": PYTORCH_SAGEMAKER_IMAGE,
            "TrainingInputMode": "File",
        },
        "HyperParameters": hyperparameters,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{S3_BUCKET}/dataset/train",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{S3_BUCKET}/dataset/val",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            },
        ],
        "OutputDataConfig": {
            "S3OutputPath": f"s3://{S3_BUCKET}/models",
        },
        "ResourceConfig": {
            "InstanceType": instance_type,
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": {
            "MaxRuntimeInSeconds": 7200,
        },
        "EnableNetworkIsolation": False,
    }

    # Use spot unless it's a GPU instance (they often have quota issues with spot)
    use_spot = "g4dn" not in instance_type and "p3" not in instance_type
    if use_spot:
        training_params["EnableManagedSpotTraining"] = True
        training_params["StoppingCondition"]["MaxWaitTimeInSeconds"] = 10800

    try:
        sm.create_training_job(**training_params)
        log_event("phase", "train", f"SageMaker job submitted: {job_name} on {instance_type}")
    except Exception as exc:
        log_event("phase", "error", f"Failed to submit SageMaker job: {exc}")
        return None

    # Poll until terminal state
    poll_interval = 60  # seconds
    while True:
        time.sleep(poll_interval)
        try:
            resp = sm.describe_training_job(TrainingJobName=job_name)
        except Exception as exc:
            log_event("phase", "warn", f"Error polling SageMaker job: {exc}")
            continue

        job_status = resp["TrainingJobStatus"]
        secondary = resp.get("SecondaryStatus", "")
        log_event("phase", "train", f"SageMaker job {job_name}: {job_status} / {secondary}")

        if job_status == "Completed":
            break
        if job_status in {"Failed", "Stopped"}:
            reason = resp.get("FailureReason", "Unknown")
            log_event("phase", "failed", f"SageMaker job {job_status}: {reason}")
            return None

    # Download output metrics
    output_key = f"models/{job_name}/output/output.tar.gz"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=output_key)
        buf = io.BytesIO(obj["Body"].read())
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            try:
                member = tar.getmember("metrics.json")
                f = tar.extractfile(member)
                if f:
                    metrics_data = json.loads(f.read().decode("utf-8"))
                    log_event("phase", "eval", f"SageMaker metrics downloaded: {metrics_data}")
                    metrics_data["job_name"] = job_name
                    metrics_data["artifact_path"] = f"s3://{S3_BUCKET}/models/{job_name}/output/model.tar.gz"
                    return metrics_data
            except KeyError:
                log_event("phase", "warn", "metrics.json not found in output tarball")
    except Exception as exc:
        log_event("phase", "warn", f"Could not download output metrics: {exc}")

    # Job completed but no metrics — return minimal info
    return {
        "job_name": job_name,
        "artifact_path": f"s3://{S3_BUCKET}/models/{job_name}/output/model.tar.gz",
        "final_val_accuracy": None,
        "final_macro_f1": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_latest_queued() -> int | None:
    engine = create_engine(DATABASE_URL)

    # Step 1: Pick queued run (read-only, auto-commit in 1.x)
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, config_id FROM training_runs "
                "WHERE status = 'queued' ORDER BY id ASC LIMIT 1"
            )
        ).fetchone()

    if row is None:
        print("[pipeline] No queued runs found.")
        return None

    run = _row_to_dict(row, ["id", "config_id"])
    run_id = run["id"]
    config_id = run["config_id"]
    print(f"[pipeline] Picked run #{run_id} (config_id={config_id})")

    run_dir = ARTIFACTS_ROOT / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    run_log_path = run_dir / "run.log"
    events_path = run_dir / "events.jsonl"

    def log_line(message: str) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        formatted = f"[{timestamp}] {message}"
        with run_log_path.open("a", encoding="utf-8") as handler:
            handler.write(formatted + "\n")
        print(message)

    def log_event(kind: str, phase: str, message: str, data: dict | None = None) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "kind": kind,
            "phase": phase,
            "message": message,
            "data": data or {},
        }
        with events_path.open("a", encoding="utf-8") as handler:
            handler.write(json.dumps(payload, ensure_ascii=True) + "\n")
        log_line(message)

    # Step 2: Mark running
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE training_runs SET status = 'running', started_at = :now, log_path = :log_path "
                "WHERE id = :id"
            ),
            {"id": run_id, "now": datetime.utcnow(), "log_path": str(run_log_path).replace("\\", "/")},
        )
    log_event("phase", "queue", f"Run #{run_id} marcada como running.")

    # Step 3: Load config
    with engine.connect() as conn:
        cfg_row = conn.execute(
            text(
                "SELECT image_size, batch_size, epochs, learning_rate, dropout, optimizer, "
                "weight_decay, scheduler, early_stopping_patience, gradient_clip, "
                "execution_target, instance_type, log_interval, auto_destroy_infra "
                "FROM training_configs WHERE id = :id"
            ),
            {"id": config_id},
        ).fetchone()
    if cfg_row:
        cfg = _row_to_dict(
            cfg_row,
            [
                "image_size",
                "batch_size",
                "epochs",
                "learning_rate",
                "dropout",
                "optimizer",
                "weight_decay",
                "scheduler",
                "early_stopping_patience",
                "gradient_clip",
                "execution_target",
                "instance_type",
                "log_interval",
                "auto_destroy_infra",
            ],
        )
    else:
        cfg = {
            "image_size": 224,
            "batch_size": 16,
            "epochs": 12,
            "learning_rate": 0.0007,
            "dropout": 0.25,
            "optimizer": "adam",
            "weight_decay": 0.0,
            "scheduler": "none",
            "early_stopping_patience": 0,
            "gradient_clip": 0.0,
            "execution_target": "local-airflow",
            "instance_type": "cpu-small",
            "log_interval": 5,
            "auto_destroy_infra": False,
        }
    log_event(
        "phase",
        "config",
        (
            f"Configuracion cargada: target={cfg['execution_target']} instance={cfg['instance_type']} "
            f"epochs={cfg['epochs']} batch={cfg['batch_size']} lr={cfg['learning_rate']}"
        ),
        cfg,
    )

    def ensure_not_cancelled() -> None:
        cancel_requested, cancel_reason = _cancellation_state(engine, run_id)
        if cancel_requested:
            raise TrainingCancelled(cancel_reason)

    def cancel_requested() -> bool:
        requested, _ = _cancellation_state(engine, run_id)
        return requested

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Train
    # ─────────────────────────────────────────────────────────────────────────
    ensure_not_cancelled()
    execution_target = cfg.get("execution_target", "local-airflow")

    if execution_target == "sagemaker":
        # ── SageMaker path ────────────────────────────────────────────────────
        log_event("phase", "train", "Lanzando entrenamiento en AWS SageMaker...")
        sm_result = _run_sagemaker(run_id, cfg, log_event)

        if sm_result is None:
            # _run_sagemaker already logged the error; mark run as failed
            with engine.begin() as conn:
                conn.execute(
                    text("UPDATE training_runs SET status = 'failed', finished_at = :now, "
                         "notes = 'SageMaker job failed. Ver events.jsonl para detalles.' WHERE id = :id"),
                    {"id": run_id, "now": datetime.utcnow()},
                )
            return run_id

        # Persist SageMaker results
        val_acc = sm_result.get("final_val_accuracy")
        macro_f1 = sm_result.get("final_macro_f1")
        artifact_path = sm_result.get("artifact_path", "")
        job_name = sm_result.get("job_name", "")
        notes = (
            f"SageMaker job: {job_name}. "
            f"val_acc={val_acc} f1={macro_f1}. Promotion metric: {PROMOTION_METRIC}."
        )

        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE training_runs SET "
                    "val_accuracy = :val_acc, test_accuracy = :test_acc, macro_f1 = :f1, "
                    "artifact_path = :artifact, log_path = :log_path, status = 'completed', "
                    "finished_at = :now, notes = :notes "
                    "WHERE id = :id"
                ),
                {
                    "id": run_id,
                    "val_acc": val_acc,
                    "test_acc": val_acc,
                    "f1": macro_f1,
                    "artifact": artifact_path,
                    "log_path": str(run_log_path).replace("\\", "/"),
                    "now": datetime.utcnow(),
                    "notes": notes,
                },
            )
            if macro_f1 is not None and val_acc is not None:
                promote_if_better(conn, run_id, macro_f1, val_acc)

        log_event("phase", "promote", f"Run #{run_id} (SageMaker) completada y persistida.")
        return run_id

    # ── Local Airflow path ────────────────────────────────────────────────────
    log_event("phase", "train", f"Iniciando entrenamiento local sobre {DATASET_ROOT}")

    progress_log = []

    def on_epoch_complete(epoch_data):
        progress_log.append(epoch_data)
        progress_path.write_text(json.dumps(progress_log, indent=2), encoding="utf-8")
        log_event(
            "epoch",
            "train",
            (
                f"Epoch {epoch_data.get('epoch')}/{epoch_data.get('total_epochs')} completada: "
                f"loss={epoch_data.get('train_loss')} acc={epoch_data.get('train_accuracy')} "
                f"val_acc={epoch_data.get('val_accuracy')}"
            ),
            epoch_data,
        )

    def on_batch_progress(batch_data):
        log_event(
            "batch",
            "train",
            (
                f"Epoch {batch_data.get('epoch')}/{batch_data.get('total_epochs')} · "
                f"batch {batch_data.get('batch')}/{batch_data.get('total_batches')} · "
                f"loss={batch_data.get('train_loss')} · acc={batch_data.get('train_accuracy')}"
            ),
            batch_data,
        )

    try:
        metrics = execute_training(
            dataset_root=DATASET_ROOT,
            output_dir=run_dir,
            image_size=cfg["image_size"],
            batch_size=cfg["batch_size"],
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            optimizer_name=cfg["optimizer"],
            dropout=cfg["dropout"],
            weight_decay=cfg["weight_decay"],
            scheduler_name=cfg["scheduler"],
            early_stopping_patience=cfg["early_stopping_patience"],
            gradient_clip=cfg["gradient_clip"],
            seed=run_id,
            feedback_root=FEEDBACK_ROOT,
            log_interval=cfg["log_interval"],
            progress_callback=on_epoch_complete,
            batch_progress_callback=on_batch_progress,
            cancel_callback=cancel_requested,
        )
        ensure_not_cancelled()
    except TrainingCancelled as exc:
        log_event("phase", "cancel", f"Training CANCELLED: {exc}", {"reason": str(exc)})
        with engine.begin() as conn:
            conn.execute(
                text(
                    "UPDATE training_runs SET status = 'cancelled', finished_at = :now, "
                    "cancelled_at = :now, cancellation_reason = :reason, notes = :notes WHERE id = :id"
                ),
                {
                    "id": run_id,
                    "now": datetime.utcnow(),
                    "reason": str(exc)[:500],
                    "notes": str(exc)[:500],
                },
            )
        return run_id
    except Exception as exc:
        log_event("phase", "failed", f"Training FAILED: {exc}", {"error": str(exc)})
        with engine.begin() as conn:
            conn.execute(
                text("UPDATE training_runs SET status = 'failed', finished_at = :now, notes = :notes WHERE id = :id"),
                {"id": run_id, "now": datetime.utcnow(), "notes": str(exc)[:500]},
            )
        raise

    log_event(
        "phase",
        "eval",
        f"Training finalizado: mode={metrics.mode} f1={metrics.macro_f1} acc={metrics.test_accuracy}",
        asdict(metrics),
    )

    # Step 6: Save results + Step 7: Promote if better
    artifact_path = str(run_dir).replace("\\", "/")
    notes = (
        f"Pipeline ejecutado automaticamente en modo {metrics.mode}. "
        f"Promotion metric: {PROMOTION_METRIC}."
    )
    with engine.begin() as conn:
        conn.execute(
            text(
                "UPDATE training_runs SET "
                "train_accuracy = :train_acc, val_accuracy = :val_acc, "
                "test_accuracy = :test_acc, macro_f1 = :f1, loss = :loss, "
                "artifact_path = :artifact, log_path = :log_path, status = 'completed', "
                "finished_at = :now, notes = :notes "
                "WHERE id = :id"
            ),
            {
                "id": run_id,
                "train_acc": metrics.train_accuracy,
                "val_acc": metrics.val_accuracy,
                "test_acc": metrics.test_accuracy,
                "f1": metrics.macro_f1,
                "loss": metrics.loss,
                "artifact": artifact_path,
                "log_path": str(run_log_path).replace("\\", "/"),
                "now": datetime.utcnow(),
                "notes": notes,
            },
        )
        promote_if_better(conn, run_id, metrics.macro_f1 or 0.0, metrics.test_accuracy or 0.0)

    log_event("phase", "promote", f"Run #{run_id} completada y persistida.")
    return run_id


if __name__ == "__main__":
    executed_run = run_latest_queued()
    if executed_run is None:
        print("No queued runs found.")
    else:
        print(f"Finished run {executed_run}.")
