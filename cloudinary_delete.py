# /backend/cloudinary_delete.py
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv
import os

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)

def delete_asset(public_id: str, resource_type: str = "image"):
    """
    Deletes asset from cloudinary.
    resource_type can be: image / video / raw
    """
    res = cloudinary.uploader.destroy(public_id, resource_type=resource_type)
    return res
