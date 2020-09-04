"""Kubeflow example using TFX DSL for local deployments (not GCP Cloud AI)."""

import os
import sys

from typing import Text

import absl
from kfp import onprem
from kfp import aws
from tfx.orchestration import pipeline
from tfx.orchestration.kubeflow import kubeflow_dag_runner

pipeline_name = "consumer_complaint_pipeline_kubeflow"

persistent_volume_claim = "efs-claim"
persistent_volume = "efs-pv"
persistent_volume_mount = "/tfx-data"
# bucket = 's3://cv-siim-isic-melanoma-classification-2020/pipeline-demo/'

# temp yaml file for Kubeflow Pipelines
output_filename = f"{pipeline_name}.yaml"
output_dir = os.path.join(
    os.getcwd(), "pipelines", "kubeflow_pipelines", "argo_pipeline_files"
)

# pipeline inputs
data_dir = os.path.join(persistent_volume_mount, "data")
# data_dir = os.path.join(bucket, "data", "/*")
module_file = os.path.join(persistent_volume_mount, "components", "module.py")
# module_file = os.path.join(bucket, "components", "module.py")

# pipeline outputs
output_base = os.path.join(persistent_volume_mount, "output")
# output_base = os.path.join(bucket, "output")
serving_model_dir = os.path.join(output_base, pipeline_name)


def init_kubeflow_pipeline(
    components, pipeline_root: Text, direct_num_workers: int
) -> pipeline.Pipeline:

    absl.logging.info(f"Pipeline root set to: {pipeline_root}")
    beam_arg = [f"--direct_num_workers={direct_num_workers}"]
    p = pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_arg,
    )
    return p


if __name__ == "__main__":

    absl.logging.set_verbosity(absl.logging.INFO)

    module_path = os.getcwd()
    if module_path not in sys.path:
        sys.path.append(module_path)

    metadata_config = kubeflow_dag_runner.get_default_kubeflow_metadata_config()
    tfx_image = os.environ.get(
        "KUBEFLOW_TFX_IMAGE",
        # "gcr.io/oreilly-book/ml-pipelines-tfx-custom:0.22.0",
        # "527798164940.dkr.ecr.us-west-2.amazonaws.com/tensorflow-2.1.0-notebook-cpu:1.1.0",
        # "tensorflow/tfx:0.22.0",
        "344971165627.dkr.ecr.us-east-1.amazonaws.com/pipeline-demo:0.3",
    )

    from pipelines.base_pipeline import init_components

    components = init_components(
        data_dir,
        module_file,
        serving_model_dir=serving_model_dir,
        training_steps=100,
        eval_steps=100,
    )

    runner_config = kubeflow_dag_runner.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        # Specify custom docker image to use.
        tfx_image=tfx_image,
        pipeline_operator_funcs=(
            # If running on K8s Engine (GKE) on Google Cloud Platform (GCP),
            # kubeflow_dag_runner.get_default_pipeline_operator_funcs()
            # provides default configurations specifically for GKE on GCP,
            # such as secrets.
            kubeflow_dag_runner.get_default_pipeline_operator_funcs()
            + [
                aws.use_aws_secret(
                    "aws-secret", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
                ),
                onprem.mount_pvc(
                    persistent_volume_claim,
                    persistent_volume,
                    persistent_volume_mount,
                ),
            ]
        ),
    )

    p = init_kubeflow_pipeline(components, output_base, direct_num_workers=0)
    output_filename = f"{pipeline_name}.yaml"
    kubeflow_dag_runner.KubeflowDagRunner(
        config=runner_config,
        output_dir=output_dir,
        output_filename=output_filename,
    ).run(p)
