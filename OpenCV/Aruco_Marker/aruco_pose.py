import cv2
import cv2.aruco as aruco
import numpy as np

cap = cv2.VideoCapture(0)

aruco_dicts = {
    "4x4": aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
    "5x5": aruco.getPredefinedDictionary(aruco.DICT_5X5_50),
    "6x6": aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
}

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
camera_matrix = np.array([
    [800, 0, frame_width / 2],
    [0, 800, frame_height / 2],
    [0, 0, 1]
], dtype=float)
dist_coeffs = np.zeros((5, 1))

marker_length = 0.05

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for name, aruco_dict in aruco_dicts.items():
        parameters = aruco.DetectorParameters()
        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)
                pos_str = f"x:{tvec[0][0]:.2f} y:{tvec[0][1]:.2f} z:{tvec[0][2]:.2f}m"
                cv2.putText(frame, pos_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Aruco Pose Estimation', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
