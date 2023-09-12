from sagemaker.model import Model
from sagemaker import get_execution_role
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel
# retrieve the llm image uri
llm_image = get_huggingface_llm_image_uri(
  "huggingface",
  version="0.8.2"
)

role = get_execution_role()
hf_model_id = "tiiuae/falcon-7b-instruct" # model id from huggingface.co/models
model_name = hf_model_id.replace("/","-").replace(".","-")
endpoint_name = "sagemaker-fastchat-01"
instance_type = "ml.g5.2xlarge" # instance type to use for deployment
number_of_gpus = 1 # number of gpus to use for inference and tensor parallelism
health_check_timeout = 2400 # Increase the timeout for the health check to 5 minutes for downloading the model

llm_model = HuggingFaceModel(
      role=role,
      image_uri=llm_image,
      env={
        'HF_MODEL_ID': hf_model_id,
        # 'HF_MODEL_QUANTIZE': "bitsandbytes", # comment in to quantize
        'SM_NUM_GPUS': f"{number_of_gpus}",
        'MAX_INPUT_LENGTH': "3000",  # Max length of input text
        'MAX_TOTAL_TOKENS': "6000",  # Max length of the generation (including input text)
      },
      name=model_name
    )

llm = llm_model.deploy(
  initial_instance_count=1,
  instance_type=instance_type,
  container_startup_health_check_timeout=health_check_timeout,
  endpoint_name=endpoint_name,
)