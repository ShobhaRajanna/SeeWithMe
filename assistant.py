import asyncio
from typing import Annotated
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import ChatContext, ChatImage, ChatMessage
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero, elevenlabs
from dotenv import load_dotenv
import os
import sentry_sdk
from ultralytics import YOLO
import numpy as np
import cv2
from PIL import Image
import io
from openai import AsyncOpenAI
import base64
import google.generativeai as genai
import time

load_dotenv()

yolo_model = YOLO("yolov8n.pt")
gemini_model = genai.GenerativeModel("gemini-1.5-flash")


class AssistantFunction(agents.llm.FunctionContext):
    @agents.llm.ai_callable(
        description="Triggered when vision capabilities are needed."
    )
    async def image(
        self,
        user_msg: Annotated[
            str, agents.llm.TypeInfo(description="User request needing vision")
        ],
    ):
        print(f"[LOG] Vision function triggered with user message: {user_msg}")
        return None


async def get_video_track(room: rtc.Room):
    video_track = asyncio.Future[rtc.RemoteVideoTrack]()
    for participant in room.remote_participants.values():
        print(f"[LOG] Checking participant: {participant.identity}")
        for pub in participant.track_publications.values():
            if (
                pub.track
                and pub.track.kind == rtc.TrackKind.KIND_VIDEO
                and isinstance(pub.track, rtc.RemoteVideoTrack)
            ):
                print(f"[LOG] Found existing video track: {pub.track.sid}")
                video_track.set_result(pub.track)
                return await video_track

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if (
            not video_track.done()
            and track.kind == rtc.TrackKind.KIND_VIDEO
            and isinstance(track, rtc.RemoteVideoTrack)
        ):
            print(f"[LOG] Subscribed to track: {track.sid}")
            video_track.set_result(track)

    try:
        return await asyncio.wait_for(video_track, timeout=10.0)
    except asyncio.TimeoutError:
        print("[ERROR] Timeout waiting for video track.")
        raise Exception("No video track received.")


async def _enableCamera(ctx):
    await ctx.room.local_participant.publish_data(
        "camera_enable", reliable=True, topic="camera"
    )


async def _getVideoFrame(ctx, assistant):
    await _enableCamera(ctx)
    latest_images_deque = []
    try:
        print("[LOG] Waiting for video track...")
        video_track = await get_video_track(ctx.room)
        print(f"[LOG] Got video track: {video_track.sid}")

        async for event in rtc.VideoStream(video_track):
            latest_image = event.frame
            latest_images_deque.append(latest_image)
            assistant.fnc_ctx.latest_image = latest_image

            if len(latest_images_deque) == 5:
                if all(is_image_too_close(f) for f in latest_images_deque):
                    await assistant.say(
                        "Please move the object a little farther from the camera for a better view.",
                        allow_interruptions=True,
                    )
                    latest_images_deque.clear()
                    return None
                if all(is_image_too_dark(f) for f in latest_images_deque):
                    await assistant.say(
                        "The image seems too dark. Could you turn on more lights?",
                        allow_interruptions=True,
                    )
                    latest_images_deque.clear()
                    return None

                best_frame = await select_best_frame(latest_images_deque)
                assistant.fnc_ctx.person_detected = await detect_person(best_frame)
                latest_images_deque.clear()  # memory
                return best_frame
    except Exception as e:
        print(f"[ERROR] Error fetching video frame: {e}")
        return None


def is_image_too_close(frame: rtc.VideoFrame) -> bool:
    try:
        img = video_frame_to_numpy(frame)
        if img is None:
            return False
        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
        print(f"[DEBUG] Frame sharpness (variance): {laplacian_var:.2f}")
        return laplacian_var < 30  # Very blurry if less than 30 variance
    except Exception as e:
        print(f"[ERROR] in is_image_too_close: {e}")
        return False


def is_image_too_dark(frame: rtc.VideoFrame) -> bool:
    try:
        img = video_frame_to_numpy(frame)
        if img is None:
            return False
        brightness = np.mean(img)
        print(f"[DEBUG] Frame brightness: {brightness:.2f}")
        return brightness < 40  # Dark if brightness under 40
    except Exception as e:
        print(f"[ERROR] in is_image_too_dark: {e}")
        return False


def video_frame_to_numpy(frame: rtc.VideoFrame) -> np.ndarray:
    try:
        width, height = frame.width, frame.height
        yuv_data = np.frombuffer(frame.data, dtype=np.uint8)
        expected_size = width * height * 3 // 2
        if yuv_data.size != expected_size:
            raise ValueError("Unexpected frame size")
        yuv_image = yuv_data.reshape((height * 3 // 2, width))
        bgr_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR_I420)
        resized = cv2.resize(bgr_image, (320, 320))
        return resized
    except Exception as e:
        print(f"[ERROR] Frame conversion error: {e}")
        return None


async def select_best_frame(frames):
    sharpest = max(
        frames, key=lambda f: cv2.Laplacian(video_frame_to_numpy(f), cv2.CV_64F).var()
    )
    return sharpest


async def detect_person(frame) -> bool:
    print("[LOG] Running YOLOv8 person detection...")
    img = video_frame_to_numpy(frame)
    if img is None:
        return False
    results = yolo_model(img, verbose=False)
    for result in results:
        if result.names and 0 in result.boxes.cls.tolist():
            print("[LOG] Person detected!")
            return True
    print("[LOG] No person detected.")
    return False


async def gemini_infer_image(frame: rtc.VideoFrame, prompt: str) -> str:
    img_np = video_frame_to_numpy(frame)
    if img_np is None:
        return "Image processing failed."

    try:
        img_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
        img_byte_arr = io.BytesIO()
        img_pil.save(img_byte_arr, format="JPEG")

        response = gemini_model.generate_content(
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img_byte_arr.getvalue(),
                            }
                        },
                    ]
                }
            ]
        )

        return (
            response.text if hasattr(response, "text") else "No response from Gemini."
        )
    except Exception as e:
        print(f"[ERROR] Gemini Vision error: {e}")
        return None


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Ally. You are an assistant for the blind and visually impaired. Your interface with users will be voice and vision."
                    "Respond with short and concise answers. Avoid using unpronouncable punctuation or emojis."
                ),
            )
        ]
    )

    gpt = openai.LLM(model="gpt-4o")

    custom_voice = elevenlabs.Voice(
        id="21m00Tcm4TlvDq8ikWAM",
        name="Bella",
        category="premade",
        settings=elevenlabs.VoiceSettings(
            stability=0.71, similarity_boost=0.5, style=0.0, use_speaker_boost=True
        ),
    )
    elevenlabs_tts = elevenlabs.TTS(voice=custom_voice)

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=gpt,
        tts=elevenlabs_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        try:
            if use_image:
                latest_image = await _getVideoFrame(ctx, assistant)
                if latest_image:
                    if assistant.fnc_ctx.person_detected:
                        print("[LOG] Person detected. Using Gemini Vision...")
                        response_text = await gemini_infer_image(latest_image, text)
                        if not response_text:
                            print(
                                "[LOG] Gemini Vision failed. Sending polite fallback response."
                            )
                            response_text = (
                                "Sorry, I couldn't analyze the image at the moment."
                            )
                        await assistant.say(response_text, allow_interruptions=True)
                    else:
                        print("[LOG] No person detected. Using GPT-4o.")
                        chat_context.messages.append(
                            ChatMessage(
                                role="user",
                                content=[text, ChatImage(image=latest_image)],
                            )
                        )
                        stream = gpt.chat(chat_ctx=chat_context)
                        await assistant.say(stream, allow_interruptions=True)
                else:
                    print("[LOG] No image captured. Using GPT-4o.")
                    chat_context.messages.append(ChatMessage(role="user", content=text))
                    stream = gpt.chat(chat_ctx=chat_context)
                    await assistant.say(stream, allow_interruptions=True)
            else:
                chat_context.messages.append(ChatMessage(role="user", content=text))
                stream = gpt.chat(chat_ctx=chat_context)
                await assistant.say(stream, allow_interruptions=True)

        except Exception as e:
            print(f"[ERROR] in _answer: {e}")
            await assistant.say(
                "Sorry, there was an internal error.",
                allow_interruptions=True,
            )

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        if msg.message:
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        if called_functions:
            user_msg = called_functions[0].call_info.arguments.get("user_msg")
            if user_msg:
                asyncio.create_task(_answer(user_msg, use_image=True))

    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("Hi there! How can I help?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
