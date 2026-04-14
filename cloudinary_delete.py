# /backend/cloudinary_delete.py
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import os

load_dotenv()

# Config: Check if using CLOUDINARY_URL or specific keys
if os.getenv("CLOUDINARY_URL"):
    # If CLOUDINARY_URL is present, it auto-configures usually, but we can be explicit
    cloudinary.config(
        cloudinary_url=os.getenv("CLOUDINARY_URL"),
        secure=True
    )
    print("Cloudinary configured via CLOUDINARY_URL")
else:
    cloudinary.config(
        cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        api_key=os.getenv("CLOUDINARY_API_KEY"),
        api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    print("Cloudinary configured via API Keys")

def delete_asset(public_id: str, resource_type: str = "image"):
    """
    Deletes asset from cloudinary.
    """
    print(f"Attempting to delete from Cloudinary: {public_id} ({resource_type})")
    try:
        res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
        print(f"Cloudinary Delete Result: {res}")
        return res
    except Exception as e:
        print(f"Cloudinary Delete Error: {e}")
        raise e

def delete_batch(public_ids: list, resource_type: str = "image"):
    """
    Deletes multiple assets from Cloudinary using the Admin API (delete_resources).
    Processed in chunks of 100 to adhere to API limits.
    """
    if not public_ids:
        print("Batch delete called with empty list.")
        return {}

    # Explicit Trace Log
    print(f"🔥🔥 STARTING BATCH DELETE: {len(public_ids)} items ({resource_type}) 🔥🔥")
    
    chunk_size = 100
    results = {}
    
    for i in range(0, len(public_ids), chunk_size):
        chunk = public_ids[i:i + chunk_size]
        attempt = 0
        max_retries = 1
        success = False
        
        while attempt <= max_retries and not success:
            try:
                print(f"   >> Executing API Batch Call (delete_resources) for chunk {i} - {i+len(chunk)}")
                res = cloudinary.api.delete_resources(chunk, resource_type=resource_type)
                
                # Res format: {'deleted': {'id1': 'deleted', 'id2': 'not_found'}, ...}
                count_deleted = len(res.get('deleted', {}))
                print(f"   >> Cloudinary Batch Chunk Success: {count_deleted} items deleted.")
                
                if 'deleted' in res:
                    results.update(res['deleted'])
                success = True
                    
            except Exception as e:
                print(f"   !! Batch API Error (Chunk {i}, Attempt {attempt+1}): {e}")
                attempt += 1
                if attempt > max_retries:
                    # Mark all as failed if retries exhausted
                    for pid in chunk:
                        results[pid] = f"Error: {str(e)}"
                else:
                    print(f"   >> Retrying chunk {i}...")

    return results

def extract_public_id_from_url(url: str) -> str:
    """
    Extracts the public_id from a Cloudinary URL.
    Handles standard URLs like:
    https://res.cloudinary.com/demo/image/upload/v1234567890/folder/my_image.jpg
    -> folder/my_image
    """
    try:
        if "cloudinary.com" not in url:
            return None
            
        # Split by '/upload/'
        parts = url.split('/upload/')
        if len(parts) < 2:
            return None
            
        path = parts[1]
        
        # Remove version if present (v12345/...)
        # Regex or simple split? 
        # Usually starts with v<digits>/
        split_path = path.split('/')
        
        if split_path[0].startswith('v') and split_path[0][1:].isdigit():
            # Skip version segment
            public_id_with_ext = "/".join(split_path[1:])
        else:
            # No version? (uncommon but possible)
            public_id_with_ext = "/".join(split_path)
            
        # Remove extension
        public_id = public_id_with_ext.rsplit('.', 1)[0]
        
        return public_id
    except Exception as e:
        print(f"Error extracting public_id from {url}: {e}")
        return None
