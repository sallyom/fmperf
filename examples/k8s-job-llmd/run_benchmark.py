"""
Modified version of example_openshift.py to run in a Kubernetes pod.
This script assumes it's running inside a pod and uses the environment variables
provided by the job configuration.
"""

import os
import urllib3
import yaml
import logging
import json
from datetime import datetime

import kubernetes
from kubernetes import client

from fmperf import Cluster
from fmperf import LMBenchmarkWorkload
from fmperf.StackSpec import StackSpec
from fmperf.utils import run_benchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def update_workload_config(workload_spec, env_vars):
    """Update workload configuration with environment variables if provided."""
    logger.info("Updating workload configuration from environment variables")
    if 'FMPERF_BATCH_SIZE' in env_vars:
        workload_spec.batch_size = int(env_vars['FMPERF_BATCH_SIZE'])
        logger.info(f"Set batch_size to {workload_spec.batch_size}")
    if 'FMPERF_SEQUENCE_LENGTH' in env_vars:
        workload_spec.sequence_length = int(env_vars['FMPERF_SEQUENCE_LENGTH'])
        logger.info(f"Set sequence_length to {workload_spec.sequence_length}")
    if 'FMPERF_MAX_TOKENS' in env_vars:
        workload_spec.max_tokens = int(env_vars['FMPERF_MAX_TOKENS'])
        logger.info(f"Set max_tokens to {workload_spec.max_tokens}")
    if 'FMPERF_NUM_USERS_WARMUP' in env_vars:
        workload_spec.num_users_warmup = int(env_vars['FMPERF_NUM_USERS_WARMUP'])
        logger.info(f"Set num_users_warmup to {workload_spec.num_users_warmup}")
    if 'FMPERF_NUM_USERS' in env_vars:
        workload_spec.num_users = int(env_vars['FMPERF_NUM_USERS'])
        logger.info(f"Set num_users to {workload_spec.num_users}")
    if 'FMPERF_NUM_ROUNDS' in env_vars:
        workload_spec.num_rounds = int(env_vars['FMPERF_NUM_ROUNDS'])
        logger.info(f"Set num_rounds to {workload_spec.num_rounds}")
    if 'FMPERF_SYSTEM_PROMPT' in env_vars:
        workload_spec.system_prompt = int(env_vars['FMPERF_SYSTEM_PROMPT'])
        logger.info(f"Set system_prompt to {workload_spec.system_prompt}")
    if 'FMPERF_CHAT_HISTORY' in env_vars:
        workload_spec.chat_history = int(env_vars['FMPERF_CHAT_HISTORY'])
        logger.info(f"Set chat_history to {workload_spec.chat_history}")
    if 'FMPERF_ANSWER_LEN' in env_vars:
        workload_spec.answer_len = int(env_vars['FMPERF_ANSWER_LEN'])
        logger.info(f"Set answer_len to {workload_spec.answer_len}")
    if 'FMPERF_TEST_DURATION' in env_vars:
        workload_spec.test_duration = int(env_vars['FMPERF_TEST_DURATION'])
        logger.info(f"Set test_duration to {workload_spec.test_duration}")
    
    return workload_spec

def save_results_to_pvc(results, run_id):
    """Save benchmark results to the PVC-mounted directory."""
    results_dir = "/results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a unique directory for this run
    run_dir = os.path.join(results_dir, f"run_{timestamp}_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Save the results
    results_file = os.path.join(run_dir, "benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")

def main():
    logger.info("Starting benchmark run")
    env_vars = os.environ
    stack_name = env_vars.get("FMPERF_STACK_NAME", "llm-d-32b-instruct")
    stack_type = env_vars.get("FMPERF_STACK_TYPE", "llm-d")
    endpoint_url = env_vars.get("FMPERF_ENDPOINT_URL", "inference-gateway")
    workload_file = env_vars.get("FMPERF_WORKLOAD_FILE", "lmbench_llama32b_instruct.yaml")
    repetition = int(env_vars.get("FMPERF_REPETITION", "1"))
    namespace = env_vars.get("FMPERF_NAMESPACE", "fmperf")

    logger.info(f"Using configuration:")
    logger.info(f"  Stack name: {stack_name}")
    logger.info(f"  Stack type: {stack_type}")
    logger.info(f"  Endpoint URL: {endpoint_url}")
    logger.info(f"  Workload file: {workload_file}")
    logger.info(f"  Repetition: {repetition}")
    logger.info(f"  Namespace: {namespace}")

    workload_file_path = os.path.join("/app/yamls", workload_file)
    logger.info(f"Loading workload configuration from {workload_file_path}")
    workload_spec = LMBenchmarkWorkload.from_yaml(workload_file_path)
    
    logger.info("Updating workload configuration with environment variables")
    workload_spec = update_workload_config(workload_spec, env_vars)

    logger.info("Creating stack specification")
    stack_spec = StackSpec(
        name=stack_name,
        stack_type=stack_type,
        refresh_interval=300,
        endpoint_url=endpoint_url
    )

    # Initialize Kubernetes client from within pod
    logger.info("Initializing Kubernetes client")
    kubernetes.config.load_incluster_config()
    apiclient = client.ApiClient()
    cluster = Cluster(name="in-cluster", apiclient=apiclient, namespace=namespace)

    # Generate a unique run ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Starting benchmark run")
    try:
        results = run_benchmark(
            cluster=cluster,
            stack_spec=stack_spec,
            workload_spec=workload_spec,
            repetition=repetition,
        )
        
        # Save results to PVC
        save_results_to_pvc(results, run_id)
        logger.info("Benchmark run completed successfully")
        
    except Exception as e:
        logger.error(f"Benchmark run failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
