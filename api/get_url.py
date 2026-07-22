import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider


def upload_image_to_oss(local_image_path: str, object_name: str = None) -> str:
    """
    上传本地图片到阿里云 OSS，并返回可访问的图片 URL。

    参数:
        local_image_path (str): 本地图片的完整路径。
        object_name (str, optional): OSS 中的目标文件路径（如 'images/example.jpg'）。
                                     若未提供，则使用本地文件名自动构造。

    返回:
        str: 图片的公开 URL（若 Bucket 为公共读）或带签名的临时 URL（若为私有）。
    """
    # 从环境变量获取凭证
    auth = oss2.ProviderAuthV4(EnvironmentVariableCredentialsProvider())

    endpoint = 'https://oss-cn-shenzhen.aliyuncs.com'
    bucket_name = 'api-bucket'
    region = 'cn-shenzhen'

    # 创建 Bucket 对象
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)

    # 如果未指定 object_name，则使用文件名作为 object_name
    if object_name is None:
        import os
        filename = os.path.basename(local_image_path)
        object_name = f"images/{filename}"

    # 上传文件
    with open(local_image_path, 'rb') as fileobj:
        bucket.put_object(object_name, fileobj)

    # 判断 Bucket ACL 权限并生成 URL
    try:
        acl = bucket.get_bucket_acl().acl
        is_public_read = (acl == oss2.BUCKET_ACL_PUBLIC_READ)
    except oss2.exceptions.AccessDenied:
        # 若无权限获取 ACL，默认当作私有处理
        is_public_read = False

    if is_public_read:
        # 构造公开 URL（注意 endpoint 不能带协议头）
        base_domain = endpoint.replace('https://', '').rstrip('/')
        image_url = f"https://{bucket_name}.{base_domain}/{object_name}"
    else:
        # 生成带签名的临时 URL（有效期 1 小时）
        expires = 3600
        image_url = bucket.sign_url('GET', object_name, expires)

    return image_url


if __name__ == "__main__":
    url = upload_image_to_oss("/data2/wangchangmiao/yjj/test_AIStation_api/edit_s0034_3d_gray.jpg")
    print(url)
