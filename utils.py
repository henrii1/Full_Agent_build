import os
import tempfile


def api_keys(openai_api_key: str = None, serper_api_key: str | None= None, 
                browserless_api_key: str | None = None ) -> dict:
    api_keys = {
        "OPENAI_API_KEY": openai_api_key,
        "SERPER_API_KEY": serper_api_key if serper_api_key else "",
        "BROWSERLESS_API_KEY": browserless_api_key if browserless_api_key else ""
    }
    return api_keys


def uploaded_files(upload_one=None, upload_two=None) -> list:
    temp_dir = tempfile.mkdtemp()
    result_list = [temp_dir]

    files = [upload_one, upload_two]

    for file in files:
        if file:  # Check if file is provided
            file_path = os.path.join(temp_dir, file.filename)
            file.save(file_path)
            result_list.append(file_path)

    return result_list

        
