import os
import time
import json
import base64
import mimetypes
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from io import BytesIO
import random
import requests

import yaml
from openai import OpenAI, AsyncOpenAI

# from util.text_render import add_text_to_image
from PIL import Image, ImageDraw, ImageFont
import os


from PIL import Image

# def compress_image(image, max_size=800):
#     if isinstance(image, str):
#         image = Image.open(image)
#     if max(image.size) > max_size:
#         ratio = max_size / max(image.size)
#         new_size = (int(image.width * ratio), int(image.height * ratio))
#         image = image.resize(new_size)
#     return image



def add_text_to_image(image_input, text, height_ratio=0.1, font_size=None):
    """
    在图片下方添加文字区域
    
    Args:
        image_input: 图片路径(str)或PIL Image对象
        text: 要添加的文字内容(str)
        height_ratio: 扩展高度占原图高度的比例(float, 默认0.1即10%)
        font_size: 字体大小(int, 如果为None则自动计算)
    
    Returns:
        PIL Image对象: 添加文字后的图片
    """
    try:
        # 处理输入，支持路径或Image对象
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            original_image = Image.open(image_input)
        elif isinstance(image_input, Image.Image):
            original_image = image_input.copy()
        else:
            raise TypeError("输入必须是图片路径(str)或PIL Image对象")
        
        # 确保图片是RGB模式
        if original_image.mode != 'RGB':
            original_image = original_image.convert('RGB')
        
        # 获取原图尺寸
        original_width, original_height = original_image.size
        
        # 计算扩展区域的高度
        text_area_height = int(original_height * height_ratio)
        
        # 创建新图片（原图高度 + 文字区域高度）
        new_height = original_height + text_area_height
        new_image = Image.new('RGB', (original_width, new_height), 'white')
        
        # 将原图粘贴到新图片的上部
        new_image.paste(original_image, (0, 0))
        
        # 创建绘图对象
        draw = ImageDraw.Draw(new_image)
        
        # 尝试加载系统默认字体
        try:
            # Windows系统
            if os.name == 'nt':
                font_path = "C:/Windows/Fonts/arial.ttf"
            # macOS系统
            elif os.uname().sysname == 'Darwin':
                font_path = "/Library/Fonts/Arial.ttf"
            # Linux系统
            else:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            
            # 如果没有指定字体大小，自动计算合适的大小
            if font_size is None:
                font_size = max(16, min(text_area_height // 2, original_width // len(text) if text else 20))
            
            font = ImageFont.truetype(font_path, font_size)
        except (OSError, IOError):
            # 如果无法加载TrueType字体，使用默认字体
            font = ImageFont.load_default()
            print("警告: 无法加载系统字体，使用默认字体")
        
        # 计算文字位置（居中显示）
        if hasattr(draw, 'textbbox'):
            # 新版本PIL
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # 兼容旧版本PIL
            text_width, text_height = draw.textsize(text, font=font)
        
        # 计算文字的起始位置（水平和垂直居中）
        x = (original_width - text_width) // 2
        y = original_height + (text_area_height - text_height) // 2
        
        # 绘制文字（黑色）
        draw.text((x, y), text, fill='black', font=font)
        
        return new_image
    
    except Exception as e:
        print(f"⚠️ 跳过图片 {image_input}，原因: {e}")
        return None


class double_image_vlm:
    def __init__(
        self,
        config_path: str = "./api/config.yaml",
        model_name: Optional[str] = "qwen3-vl-235b-a22b-instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        基于 AiHubMix 的 Qwen VL 客户端
        - 读取配置文件
        - 创建 OpenAI 客户端
        - 支持重试、图片输入与文本输出解析
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        # 配置优先级: 显式参数 > 配置文件 > 默认
        self.api_key = api_key or self.config.get("api_key") or ""
        self.base_url = base_url or self.config.get("base_url") or "https://aihubmix.com/v1"
        self.model_name = model_name or self.config.get("model_name") or "qwen3-vl-235b-a22b-instruct"
        self.temperature = self.config.get("temperature", 0.3)

        self.max_retries = int(self.config.get("max_retries", 1))
        self.retry_delay = float(self.config.get("retry_delay", 2))

        if not self.api_key:
            raise ValueError("缺少 api_key，请在 ./api/config.yaml 中设置 api_key")

        # 站点信息（可选，用于统计/溯源）
        self.site_url = self.config.get("site_url", "")
        self.site_name = self.config.get("site_name", "")

        # 同步和异步客户端
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        print(f"[{self.model_name}] 初始化完成，模型: {self.model_name}，网关: {self.base_url}")

    @staticmethod
    def encode_image(image_input) -> str:
        """将本地图片路径或图片对象编码为 base64 字符串
        
        Args:
            image_input: 支持以下类型:
                - str: 图片文件路径
                - PIL.Image.Image: PIL图片对象
                - bytes: 图片二进制数据
                - numpy.ndarray: numpy图片数组
        
        Returns:
            str: base64编码的图片字符串
        """
        
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")
        elif hasattr(image_input, 'save') and hasattr(image_input, 'format'):
            buffer = BytesIO()
            format = image_input.format or 'PNG'
            image_input.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")        
        else:
            raise TypeError(f"不支持的图片输入类型: {type(image_input)}")

    # async def generate_with_image_async(
    #     self,
    #     prompt: str,
    #     input_image_path: str,
    #     output_image_path: str,
    #     input_image_lable: str = "Input",
    #     output_image_lable: str = "Output",
    #     temperature: Optional[float] = None,
    #     max_tokens: Optional[int] = None,
    # ) -> Dict[str, Any]:
    #     temperature = temperature if temperature is not None else self.temperature
    #     max_tokens = max_tokens or 4096
        
    #     input_image = add_text_to_image(input_image_path, input_image_lable)
    #     output_image = add_text_to_image(output_image_path, output_image_lable)

    #     input_img_base64 = self.encode_image(input_image)
    #     output_img_base64 = self.encode_image(output_image)
        
    #     input_img_data_url = f"data:image/png;base64,{input_img_base64}"
    #     output_img_data_url = f"data:image/png;base64,{output_img_base64}"

    #     messages = [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {"type": "image_url", "image_url": {"url": input_img_data_url}},
    #                 {"type": "image_url", "image_url": {"url": output_img_data_url}},
    #             ],
    #         }
    #     ]

    #     # 准备可选 headers
    #     extra_headers = {}
    #     if self.site_url:
    #         extra_headers["HTTP-Referer"] = self.site_url
    #     if self.site_name:
    #         extra_headers["X-Title"] = self.site_name

    #     retries = 0
    #     last_err = None

    #     while retries < self.max_retries:
    #         try:
    #             response = await self.async_client.chat.completions.create(
    #                 model=self.model_name,
    #                 messages=messages,
    #                 temperature=temperature,
    #                 max_tokens=max_tokens,
    #                 stream=False,  # 不使用流式输出
    #                 extra_headers=extra_headers if extra_headers else None,
    #             )

    #             return self._parse_response(response)

    #         except Exception as e:
    #             retries += 1
    #             last_err = e
    #             print(f"[ERROR] 异步调用失败({retries}/{self.max_retries}): {e}")
    #             if retries >= self.max_retries:
    #                 break
    #             await asyncio.sleep(self.retry_delay * (2 ** (retries - 1)))

    #     return {"error": f"调用失败: {last_err}", "text": "", "usage": {}}

    async def generate_with_image_async(
        self,
        prompt: str,
        input_image_path: str = "",
        output_image_path: str = "",
        input_image_lable: str = "Input",
        output_image_lable: str = "Output",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or 4096

        content_parts = [{"type": "text", "text": prompt}]
        try:
        # 处理输入图像（如果提供）
            if input_image_path.strip():
                input_image = add_text_to_image(input_image_path, input_image_lable)
                input_img_base64 = self.encode_image(input_image)
                input_img_data_url = f"data:image/png;base64,{input_img_base64}"
                content_parts.append({"type": "image_url", "image_url": {"url": input_img_data_url}})
        except Exception as e:
            print(f"⚠️ 跳过图片 {input_image_path}，原因: {e}")

        # 处理输出图像（如果提供）
        if output_image_path.strip():
            output_image = add_text_to_image(output_image_path, output_image_lable)
            output_img_base64 = self.encode_image(output_image)
            output_img_data_url = f"data:image/png;base64,{output_img_base64}"
            content_parts.append({"type": "image_url", "image_url": {"url": output_img_data_url}})

        messages = [{"role": "user", "content": content_parts}]

        # 准备可选 headers
        extra_headers = {}
        if self.site_url:
            extra_headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            extra_headers["X-Title"] = self.site_name

        retries = 0
        last_err = None

        while retries < self.max_retries:
            try:
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    extra_headers=extra_headers if extra_headers else None,
                )
                return self._parse_response(response)

            except Exception as e:
                retries += 1
                last_err = e
                print(f"[ERROR] 异步调用失败({retries}/{self.max_retries}): {e}")
                if retries >= self.max_retries:
                    break
                await asyncio.sleep(self.retry_delay * (2 ** (retries - 1)))

        return {"error": f"调用失败: {last_err}", "text": "", "usage": {}}

    def _parse_response(self, response) -> Dict[str, Any]:
        """解析响应的通用方法"""
        # 解析文本内容
        text_content = ""
        
        msg = response.choices[0].message
        
        # 从 content 中提取文本
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                text_content = msg.content.strip()
            elif isinstance(msg.content, list):
                # 兼容某些返回结构
                text_parts = []
                for c in msg.content:
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                        text_parts.append(c["text"])
                    elif isinstance(c, str):
                        text_parts.append(c)
                text_content = "".join(text_parts).strip()

        # 解析 usage 信息
        usage = {}
        if hasattr(response, "usage") and response.usage:
            try:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            except Exception:
                usage = {}

        print(f"[{self.model_name}] created: {getattr(response, 'created', '')}, usage: {usage}")

        return {
            "text": text_content,
            "usage": usage,
            "raw": response,
        }

    async def generate_batch(
        self,
        requests: List[Tuple[str, str, str, Optional[str], Optional[str], Optional[float], Optional[int]]],
        concurrency: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个请求
        
        Args:
            requests: List of (prompt, image_path, temperature, max_tokens)
            concurrency: 并发数量
            
        Returns:
            List of response dicts, 保持与输入相同的顺序
        """
        sem = asyncio.Semaphore(concurrency)

        async def _process_one(idx: int, request_args: Tuple):
            async with sem:
                prompt, input_image_path, output_image_path, input_image_lable, output_image_lable, temperature, max_tokens = request_args
                try:
                    result = await self.generate_with_image_async(
                        prompt=prompt,
                        input_image_path=input_image_path,
                        output_image_path=output_image_path,
                        input_image_lable = input_image_lable,
                        output_image_lable = output_image_lable,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                except Exception as e:
                    print(f"⚠️ 跳过图片 {input_image_path}，原因: {e}")

                return idx, result

        # 创建所有异步任务
        tasks = [
            asyncio.create_task(_process_one(i, request))
            for i, request in enumerate(requests)
        ]

        # 收集结果并保持顺序
        results = [None] * len(tasks)
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result

        return results


class single_image_vlm:
    def __init__(
        self,
        config_path: str = "./api/config.yaml",
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        基于 AiHubMix 的 Qwen VL 客户端
        - 读取配置文件
        - 创建 OpenAI 客户端
        - 支持重试、图片输入与文本输出解析
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)

        if model_name == "Ming-UniVision_VLM4gen" or model_name == "Ming-UniVision_VLM" or model_name == "HuatuoGPT-Vision" or model_name == "RadFM" or model_name == "Showo_VLM" or model_name == "Showo_VLM4EDIT" :
            self.temperature = self.config.get("temperature", 0.3)

            self.max_retries = int(self.config.get("max_retries", 3))
            self.retry_delay = float(self.config.get("retry_delay", 2))

            self.model_name = model_name

            print(f"[{self.model_name}] 初始化完成，模型: {self.model_name}")

        else:
            # 配置优先级: 显式参数 > 配置文件 > 默认
            self.api_key = api_key or self.config.get("api_key") or ""
            self.base_url = base_url or self.config.get("base_url") or "https://aihubmix.com"
            self.model_name = model_name or self.config.get("model_name") or (_ for _ in ()).throw(ValueError("❌ 未指定模型名称"))
            self.temperature = self.config.get("temperature", 0.3)

            self.max_retries = int(self.config.get("max_retries", 3))
            self.retry_delay = float(self.config.get("retry_delay", 2))

            if not self.api_key:
                raise ValueError("缺少 api_key，请在 ./api/config.yaml 中设置 api_key")

            # 站点信息（可选，用于统计/溯源）
            self.site_url = self.config.get("site_url", "")
            self.site_name = self.config.get("site_name", "")

            # 同步和异步客户端
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            self.async_client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            print(f"[{self.model_name}] 初始化完成，模型: {self.model_name}，网关: {self.base_url}")

    @staticmethod
    def encode_image(image_input) -> str:
        """将本地图片路径或图片对象编码为 base64 字符串
        
        Args:
            image_input: 支持以下类型:
                - str: 图片文件路径
                - PIL.Image.Image: PIL图片对象
                - bytes: 图片二进制数据
                - numpy.ndarray: numpy图片数组
        
        Returns:
            str: base64编码的图片字符串
        """
        
        # image_input = compress_image(image_input)

        
        if isinstance(image_input, str):
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"图片文件不存在: {image_input}")
            with open(image_input, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        elif isinstance(image_input, bytes):
            return base64.b64encode(image_input).decode("utf-8")
        elif hasattr(image_input, 'save') and hasattr(image_input, 'format'):
            buffer = BytesIO()
            format = image_input.format or 'PNG'
            image_input.save(buffer, format=format)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")        
        else:
            raise TypeError(f"不支持的图片输入类型: {type(image_input)}")

    @staticmethod
    def guess_mime_type(image_path: str) -> str:
        """根据文件后缀猜测 MIME 类型，默认 image/png"""
        mime, _ = mimetypes.guess_type(image_path)
        if mime and mime.startswith('image/'):
            return mime
        return "image/png"

    async def generate_with_image_async(
        self,
        prompt: str,
        image_path: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.model_name == "Ming-UniVision_VLM4gen" or self.model_name == "Ming-UniVision_VLM" or self.model_name == "HuatuoGPT-Vision" or self.model_name == "RadFM" or self.model_name == "Showo_VLM" or self.model_name == "Showo_VLM4EDIT":
            import requests, base64, types, json

            # 读取图片
            with open(image_path, "rb") as f:
                image_bytes = f.read()
            input_image_base64 = base64.b64encode(image_bytes).decode("utf-8")


            if self.model_name == "Ming-UniVision_VLM4gen":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "Ming-UniVision-16B-A3B",
                    "mode": "understanding",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")
                
                response = requests.post(
                    "http://127.0.0.1:10093/api/run_model",
                    data=payload,
                    timeout=120
                )

            elif self.model_name == "Ming-UniVision_VLM":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "Ming-UniVision-16B-A3B",
                    "mode": "understanding",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")
                
                response = requests.post(
                    "http://127.0.0.1:10094/api/run_model",
                    data=payload,
                    timeout=60
                )

            elif self.model_name == "HuatuoGPT-Vision":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "HuatuoGPT-Vision-7B",
                    "mode": "VLM",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")

                response = requests.post(
                    "http://127.0.0.1:10095/api/run_model",
                    data=payload,
                    timeout=60
                )

            elif self.model_name == "RadFM":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "RadFM",
                    "mode": "VLM",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")

                response = requests.post(
                    "http://127.0.0.1:10096/api/run_model",
                    data=payload,
                    timeout=360
                )

            elif self.model_name == "Showo_VLM":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "VLM",
                    "mode": "VLM",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")

                response = requests.post(
                    "http://127.0.0.1:10097/api/run_model",
                    data=payload,
                    timeout=360
                )

            elif self.model_name == "Showo_VLM4EDIT":
                payload = {
                    "model_name": f"{self.model_name}",
                    "weight_version": "VLM",
                    "mode": "VLM",
                    "input_text": prompt,
                    "input_image_base64": input_image_base64
                }
                
                print(f"qqqqqqqqqqqqqqqq {payload['model_name']}")

                response = requests.post(
                    "http://127.0.0.1:10099/api/run_model",
                    data=payload,
                    timeout=360
                ) 
            
            else:
                raise ValueError(f"不支持的模型名称: {self.model_name}")
            

            result = response.json()

            return self._parse_local_api_response(result)

            # output_text = result.get("output_text", "")
            # generate_prompt = result.get("input_text", "")

            # # 返回的 text 必须是 JSON 字符串，和后续 extract_json 一致
            # text_content = json.dumps({
            #     "output_text": output_text,
            #     "generate_prompt": generate_prompt
            # }, ensure_ascii=False)

            # # 构造伪 raw 对象
            # msg = types.SimpleNamespace()
            # msg.content = [{"type": "text", "text": text_content}]

            # fake_response = types.SimpleNamespace()
            # fake_response.choices = [types.SimpleNamespace(message=msg)]
            # fake_response.usage = types.SimpleNamespace(
            #     prompt_tokens=None,
            #     completion_tokens=None,
            #     total_tokens=None
            # )
            # fake_response.created = None

            # return {
            #     "text": text_content,  # JSON 字符串，包含 output_text 和 generate_prompt
            #     "usage": {
            #         "prompt_tokens": None,
            #         "completion_tokens": None,
            #         "total_tokens": None
            #     },
            #     "raw": fake_response
            # }



        else:
            temperature = temperature if temperature is not None else self.temperature
            max_tokens = max_tokens or 4096

            base64_img = self.encode_image(image_path)
            mime_type = self.guess_mime_type(image_path)
            img_data_url = f"data:{mime_type};base64,{base64_img}"

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_data_url}},
                    ],
                }
            ]

            # 准备可选 headers
            extra_headers = {}
            if self.site_url:
                extra_headers["HTTP-Referer"] = self.site_url
            if self.site_name:
                extra_headers["X-Title"] = self.site_name

            retries = 0
            last_err = None

            while retries < self.max_retries:
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=False,  # 不使用流式输出
                        extra_headers=extra_headers if extra_headers else None,
                    )

                    return self._parse_response(response)

                except Exception as e:
                    retries += 1
                    last_err = e
                    print(f"[ERROR] 异步调用失败({retries}/{self.max_retries}): {e}")
                    if retries >= self.max_retries:
                        break
                    await asyncio.sleep(self.retry_delay * (2 ** (retries - 1)))
                    # await asyncio.sleep(self.retry_delay * (2 ** (retries - 1)) + random.uniform(0, 0.5))


            return {"error": f"调用失败: {last_err}", "text": "", "usage": {}}

    # def _parse_local_api_response(self, response: dict) -> Dict[str, Any]:
    #     text_content = response.get("output_text", "").strip()
    #     return {
    #         "text": text_content,
    #         "raw": response
    #     }

    def _parse_local_api_response(self, response: dict) -> dict:
        # 删除 input_image_base64 字段（如果存在）
        response.pop("input_image_base64", None)

        output = response.get("output_text", "")

        if isinstance(output, list):
            # 如果 output_text 是列表，则拼接为字符串
            text_content = " ".join(str(x) for x in output).strip()
        else:
            # 如果是字符串，则直接 strip
            text_content = str(output).strip()

        return {
            "text": text_content,
            "raw": response
        }



    def _parse_response(self, response) -> Dict[str, Any]:
        """解析响应的通用方法"""
        # 解析文本内容
        text_content = ""
        
        msg = response.choices[0].message

        # 从 content 中提取文本
        if hasattr(msg, "content") and msg.content:
            if isinstance(msg.content, str):
                text_content = msg.content.strip()
            elif isinstance(msg.content, list):
                # 兼容某些返回结构
                text_parts = []
                for c in msg.content:
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                        text_parts.append(c["text"])
                    elif isinstance(c, str):
                        text_parts.append(c)
                text_content = "".join(text_parts).strip()

        # 解析 usage 信息
        usage = {}
        if hasattr(response, "usage") and response.usage:
            try:
                usage = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }
            except Exception:
                usage = {}

        print(f"[{self.model_name}] created: {getattr(response, 'created', '')}, usage: {usage}")
        print(f"响应内容: {text_content}")
        return {
            "text": text_content,
            "usage": usage,
            "raw": response,
        }

    async def generate_batch(
        self,
        requests: List[Tuple[str, str, Optional[float], Optional[int]]],
        concurrency: int = 8,
    ) -> List[Dict[str, Any]]:
        """
        批量处理多个请求
        
        Args:
            requests: List of (prompt, image_path, temperature, max_tokens)
            concurrency: 并发数量
            
        Returns:
            List of response dicts, 保持与输入相同的顺序
        """
        sem = asyncio.Semaphore(concurrency)

        async def _process_one(idx: int, request_args: Tuple):
            async with sem:
                prompt, image_path, temperature, max_tokens = request_args
                result = await self.generate_with_image_async(
                    prompt=prompt,
                    image_path=image_path,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return idx, result

        # 创建所有异步任务
        tasks = [
            asyncio.create_task(_process_one(i, request))
            for i, request in enumerate(requests)
        ]

        # 收集结果并保持顺序
        results = [None] * len(tasks)
        for coro in asyncio.as_completed(tasks):
            idx, result = await coro
            results[idx] = result

        return results



async def demo_batch():
    """
    批量处理演示
    """
    client = single_image_vlm(config_path="./api/config.yaml",model_name="qwen3-vl-235b-a22b-instruct")
    
    # 准备批量请求
    requests = [
        ("请描述这张图片", "./output_image/0a41a27775c589d5.png", 0.7, 2048),
        ("分析这张图片的内容", "./output_image/0afbfcc1994b2600.png", 0.8, 2048),
        ("这张图片展示了什么？", "./output_image/0b0c027ecaffc52a.png", 0.6, 2048),
    ]
    
    # 批量处理
    results = await client.generate_batch(requests, concurrency=3)
    
    for i, result in enumerate(results):
        print(f"\n--- 结果 {i+1} ---")
        if "error" in result:
            print(f"错误: {result['error']}")
        else:
            print(result["text"])
            print(f"使用情况: {result['usage']}")


if __name__ == "__main__":
    
    # 批量请求演示
    asyncio.run(demo_batch())