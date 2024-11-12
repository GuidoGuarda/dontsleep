import cv2
import mediapipe as mp
import numpy as np
import time
import pygame
import os
import sys

# Inicializa o mixer de áudio
pygame.mixer.init()

# Carrega os arquivos de som
audio_suspense = pygame.mixer.Sound("suspense.mp3")
audio_grito = pygame.mixer.Sound("susto-grito.mp3")

# Pontos dos olhos e boca
p_olho_esq = [385, 380, 387, 373, 362, 263]
p_olho_dir = [160, 144, 158, 153, 33, 133]
p_olhos = p_olho_esq + p_olho_dir
p_boca = [82, 87, 13, 14, 312, 317, 78, 308]

# Função EAR
def calculo_ear(face, p_olho_dir, p_olho_esq):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_esq = face[p_olho_esq, :]
        face_dir = face[p_olho_dir, :]

        ear_esq = (np.linalg.norm(face_esq[0] - face_esq[1]) + np.linalg.norm(face_esq[2] - face_esq[3])) / (2 * (np.linalg.norm(face_esq[4] - face_esq[5])))
        ear_dir = (np.linalg.norm(face_dir[0] - face_dir[1]) + np.linalg.norm(face_dir[2] - face_dir[3])) / (2 * (np.linalg.norm(face_dir[4] - face_dir[5])))
    except:
        ear_esq = 0.0
        ear_dir = 0.0
    media_ear = (ear_esq + ear_dir) / 2
    return media_ear

# Função MAR
def calculo_mar(face, p_boca):
    try:
        face = np.array([[coord.x, coord.y] for coord in face])
        face_boca = face[p_boca, :]

        mar = (np.linalg.norm(face_boca[0] - face_boca[1]) + np.linalg.norm(face_boca[2] - face_boca[3]) + np.linalg.norm(face_boca[4] - face_boca[5])) / (2 * (np.linalg.norm(face_boca[6] - face_boca[7])))
    except:
        mar = 0.0
    return mar

# Limiares
ear_limiar = 0.27
mar_limiar = 0.1

# Inicializa a câmera
cap = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Canais para áudio
channel_suspense = pygame.mixer.Channel(0)
channel_grito = pygame.mixer.Channel(1)

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as facemesh:
    while cap.isOpened():
        sucesso, frame = cap.read()
        if not sucesso:
            print('Ignorando o frame vazio da câmera.')
            continue
        
        comprimento, largura, _ = frame.shape
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        saida_facemesh = facemesh.process(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if saida_facemesh.multi_face_landmarks:
            for face_landmarks in saida_facemesh.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(127,255,0), thickness=1, circle_radius=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 100, 0), thickness=1, circle_radius=1)
                )
                
                face = face_landmarks.landmark

                # Calcula EAR e MAR
                ear = calculo_ear(face, p_olho_dir, p_olho_esq)
                mar = calculo_mar(face, p_boca)

                # Controle do áudio suspense (olhos abertos)
                if ear >= ear_limiar:
                    if not channel_suspense.get_busy():
                        channel_suspense.play(audio_suspense, loops=-1)
                else:
                    channel_suspense.stop()

                # Controle do áudio grito (boca aberta)
                if mar >= mar_limiar:
                    if not channel_grito.get_busy():
                        channel_grito.play(audio_grito)
                else:
                    channel_grito.stop()

        cv2.imshow('Camera', frame)
        if cv2.waitKey(10) & 0xFF == ord('c'):
            break

cap.release()
cv2.destroyAllWindows()
