from google.cloud import storage

BUCKET_NAME = 'sync_testinput'

def blob_retrieval(file_name: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(file_name)
    contents = blob.download_as_string()
    return contents