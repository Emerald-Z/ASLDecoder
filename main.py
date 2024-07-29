import asyncio
import pyaudio
import numpy as np
import cv2

from viam.robot.client import RobotClient
from viam.rpc.dial import Credentials, DialOptions
from chat_service_api import Chat
from speech_service_api import SpeechService
from viam.components.camera import Camera
from viam.services.vision import VisionClient
from viam.media.video import ViamImage, CameraMimeType

from kaggleasl5thplacesolution.codesmall.inference import get_model_output

async def connect():
    opts = RobotClient.Options.with_api_key(
      api_key='h1rgrtd3j552kyg74txtrhgyybf4smyt',
      api_key_id='e4fd8151-5e29-4c1f-90c9-7581846e839a'
    )
    return await RobotClient.at_address('llm-main.ll0jxae16a.viam.cloud', opts)

def read_video_cv2(vid, n_frames=1000):
    cap = cv2.VideoCapture(vid)
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        if frame is None:
            break
        arr = np.array(frame)
        # print(arr.shape)
        all.append(arr)
        i += 1
    return np.array(all)

async def main():
    robot = await connect()
    llm = Chat.from_robot(robot, name="chat")
    speech = SpeechService.from_robot(robot, name="speech")
    camera = Camera.from_robot(robot=robot, name="cam")
    hand_model = VisionClient.from_robot(robot, "hand_class")

    # read video
    # slice frames
    # video = read_video_cv2("/Users/emerald.zhang@viam.com/Downloads/face.mov")
    video = []
    # get output from model
    classification = "no-hand"
    # start recording when hand detected
    while classification == "no-hand":
        img = await camera.get_image()
        output = await hand_model.get_classifications(img, 1)
        classification = output[0].class_name
        print(classification)
    while classification == "hand":
        print("recording")
        img = await camera.get_image()
        output = await hand_model.get_classifications(img, 1)
        classification = output[0].class_name
        image = cv2.imdecode(np.frombuffer(img.data, 'uint8'), cv2.IMREAD_COLOR)
        # image = np.transpose(image, (1, 0, 2)=]
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        video.append(image) # probably VIAM image format
    video = np.array(video)
    print(video.shape)
    words = get_model_output(video)
    print(words)
    message = f""" Important: Just return the sentence to me with no other input from you.
            Words: {words} """
    response = await llm.chat(message)
    print(response)
    # await speech.say(response, True)

    # Don't forget to close the machine when you're done!
    await robot.close()

if __name__ == '__main__':
    asyncio.run(main())

