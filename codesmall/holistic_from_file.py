import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import numpy as np
import torch
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

LHAND = np.arange(468, 489).tolist() # 21
RHAND = np.arange(522, 543).tolist() # 21
POSE  = np.arange(489, 522).tolist() # 33
FACE  = np.arange(0,468).tolist()    #468

REYE = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    246, 161, 160, 159, 158, 157, 173,
][::2]
LEYE = [
    263, 249, 390, 373, 374, 380, 381, 382, 362,
    466, 388, 387, 386, 385, 384, 398,
][::2]
NOSE=[
    1,2,98,327
]
SLIP = [
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415,
]
SPOSE = (np.array([
    11,13,15,12,14,16,23,24,
])+489).tolist()

BODY = REYE+LEYE+NOSE+SLIP+SPOSE

def get_indexs(L):
    return sorted([i + j * len(L) for i in range(len(L)) for j in range(len(L)) if i>j])

DIST_INDEX = get_indexs(RHAND)

LIP_DIST_INDEX = get_indexs(SLIP)

POSE_DIST_INDEX = get_indexs(SPOSE)

EYE_DIST_INDEX = get_indexs(REYE)

NOSE_DIST_INDEX = get_indexs(NOSE)

def pre_process(xyz):

    # select the lip, right/left hand, right/left eye, pose, nose parts.
    lip   = xyz[:, SLIP]#20
    lhand = xyz[:, LHAND]#21
    rhand = xyz[:, RHAND]#21
    pose = xyz[:, SPOSE]#8
    reye = xyz[:, REYE]#16
    leye = xyz[:, LEYE]#16
    nose = xyz[:, NOSE]#4

    xyz = torch.cat([ #(none, 106, 2)
        lhand,
        rhand,
        lip,
        pose,
        reye,
        leye,
        nose,
    ],1)


    # concatenate the frame delta information
    x = torch.cat([xyz[1:,:len(LHAND+RHAND),:]-xyz[:-1,:len(LHAND+RHAND),:],torch.zeros((1,len(LHAND+RHAND),2))],0)
    
    
    # TODO
    ld = lhand[:,:,:2].reshape(-1,len(LHAND),1,2)-lhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    ld = torch.sqrt((ld**2).sum(-1))
    ld = ld.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    rd = rhand[:,:,:2].reshape(-1,len(LHAND),1,2)-rhand[:,:,:2].reshape(-1,1,len(LHAND),2)
    rd = torch.sqrt((rd**2).sum(-1))
    rd = rd.reshape(-1,len(LHAND)*len(LHAND))[:,DIST_INDEX]
    
    lipd = lip[:,:,:2].reshape(-1,len(SLIP),1,2)-lip[:,:,:2].reshape(-1,1,len(SLIP),2)
    lipd = torch.sqrt((lipd**2).sum(-1))
    lipd = lipd.reshape(-1,len(SLIP)*len(SLIP))[:,LIP_DIST_INDEX]
    
    posed = pose[:,:,:2].reshape(-1,len(SPOSE),1,2)-pose[:,:,:2].reshape(-1,1,len(SPOSE),2)
    posed = torch.sqrt((posed**2).sum(-1))
    posed = posed.reshape(-1,len(SPOSE)*len(SPOSE))[:,POSE_DIST_INDEX]
    
    reyed = reye[:,:,:2].reshape(-1,len(REYE),1,2)-reye[:,:,:2].reshape(-1,1,len(REYE),2)
    reyed = torch.sqrt((reyed**2).sum(-1))
    reyed = reyed.reshape(-1,len(REYE)*len(REYE))[:,EYE_DIST_INDEX]
    
    leyed = leye[:,:,:2].reshape(-1,len(LEYE),1,2)-leye[:,:,:2].reshape(-1,1,len(LEYE),2)
    leyed = torch.sqrt((leyed**2).sum(-1))
    leyed = leyed.reshape(-1,len(LEYE)*len(LEYE))[:,EYE_DIST_INDEX]

    dist_hand=torch.sqrt(((lhand-rhand)**2).sum(-1))

    xyz = torch.cat([xyz.reshape(-1,(len(LHAND+RHAND+REYE+LEYE+NOSE+SLIP+SPOSE))*2), 
                         x.reshape(-1,(len(LHAND+RHAND))*2),
                         ld,
                         rd,
                         lipd,
                         posed,
                         reyed,
                         leyed,
                         dist_hand,
                        ],1)
    
    # fill the nan value with 0
    xyz[torch.isnan(xyz)] = 0

    return xyz

def reshape(data):
    # print(data.shape)
    maxlen = 256 #537 actually
    data = np.expand_dims(data, 0)
    xyz = data.reshape((-1, 543, 3))

    xyz = xyz[:, :, :2]
    xyz_flat = xyz.reshape(-1, 2)
    m = np.nanmean(xyz_flat, 0).reshape(1, 1, 2)

    xyz = torch.from_numpy(xyz).float()
    xyz = pre_process(xyz)[:maxlen]

    xyz[torch.isnan(xyz)] = 0
    data_pad = torch.zeros((maxlen, xyz.shape[1]), dtype=torch.float32)
    tot = xyz.shape[0]

    if tot <= maxlen:
        data_pad[:tot] = xyz
    else:
        data_pad[:] = xyz[:maxlen]
    # print(data_pad.shape)
    return np.expand_dims(data_pad, 0)

def get_landmarks(image, draw=False):
   # For webcam input:
    image = cv2.imread("/Users/emerald.zhang@viam.com/Downloads/asl1.jpg")
    # For static images:
    IMAGE_FILES = ["/Users/emerald.zhang@viam.com/Downloads/asl1.jpg"]
    BG_COLOR = (192, 192, 192) # gray
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            print(len(results.face_landmarks.landmark))
            f_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])[0:468]
            p_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            rh_landmarks = np.empty((21, 3))
            lh_landmarks = np.empty((21, 3))

            if results.right_hand_landmarks is not None:
                rh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            if results.left_hand_landmarks is not None:
                lh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

            print(rh_landmarks.shape, f_landmarks.shape, p_landmarks.shape)
            landmarks = np.concatenate((f_landmarks, lh_landmarks, p_landmarks, rh_landmarks), axis=0)
            # if results.pose_landmarks:
            #     print(
            #         f'Nose coordinates: ('
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].x * image_width}, '
            #         f'{results.pose_landmarks.landmark[mp_holistic.PoseLandmark.NOSE].y * image_height})'
            #     )

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the image.
            if draw:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.
                    get_default_pose_landmarks_style())
                # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
                # Plot pose world landmarks.
                mp_drawing.plot_landmarks(
                    results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        # cv2.waitKey(0)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        print(landmarks.shape)
        return reshape(landmarks)

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

def get_landmarks_from_video(video, draw=False):
    # video = read_video_cv2("/Users/emerald.zhang@viam.com/Downloads/asl1.jpg")

   # For webcam input:
    landmarks_arr = []
    # For static images:
    BG_COLOR = (192, 192, 192) # gray
    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        refine_face_landmarks=True) as holistic:
        for idx, image in enumerate(video):
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            # print(len(results.face_landmarks.landmark))
            f_landmarks = np.empty((468, 3))
            p_landmarks = np.empty((33, 3))
            rh_landmarks = np.empty((21, 3))
            lh_landmarks = np.empty((21, 3))

            if results.face_landmarks is not None:
                f_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])[0:468]
            if results.pose_landmarks is not None:
                p_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
            if results.right_hand_landmarks is not None:
                rh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
            if results.left_hand_landmarks is not None:
                lh_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])

            # print(rh_landmarks.shape, f_landmarks.shape, p_landmarks.shape)
            landmarks = np.concatenate((f_landmarks, lh_landmarks, p_landmarks, rh_landmarks), axis=0)


            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose, left and right hands, and face landmarks on the image.
            if draw:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    annotated_image,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.
                    get_default_pose_landmarks_style())
                # cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
                # Plot pose world landmarks.
                mp_drawing.plot_landmarks(
                    results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
            landmarks_arr.append(landmarks)
        # Flip the image horizontally for a selfie-view display.
        # cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        # cv2.waitKey(0)
        # if cv2.waitKey(5) & 0xFF == 27:
        #     break
        # print(landmarks.shape)
        reshaped = reshape(np.array(landmarks_arr))
        return reshaped
