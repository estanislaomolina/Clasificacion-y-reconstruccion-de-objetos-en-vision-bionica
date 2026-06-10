"""
simulator.py
Wrapper sobre pulse2percept para simular la percepción prostética.
Soporta Argus II, PRIMA y una grilla personalizada de alta resolución.
"""

from typing import Literal
import numpy as np
import cv2

ImplantType = Literal["argus2", "prima", "custom_grid"]


class ProstheticSimulator:
    """
    Simula la percepción de una imagen a través de un implante retinal.

    Parámetros
    ----------
    implant_type : ImplantType
        'argus2'      — Argus II (60 electrodos, baja resolución)
        'prima'       — PRIMA (~378 electrodos, media resolución)
        'custom_grid' — Grilla personalizada de alta resolución (cota superior)
    """

    def __init__(self, implant_type: ImplantType = "argus2"):
        self.implant_type = implant_type
        self._model = None
        self._implant = None
        self._setup()

    def _setup(self):
        import pulse2percept as p2p

        if self.implant_type == "argus2":
            self._implant = p2p.implants.ArgusII()
            self._model = p2p.models.ScoreboardModel()

        elif self.implant_type == "prima":
            self._implant = p2p.implants.PRIMA()
            self._model = p2p.models.ScoreboardModel()

        elif self.implant_type == "custom_grid":
            # Grilla de alta resolución: 30x30 electrodos, separación 100 µm
            self._implant = p2p.implants.ProsthesisSystem(
                earray=p2p.implants.ElectrodeGrid(
                    (30, 30), 100, names=("A", "1")
                )
            )
            self._model = p2p.models.ScoreboardModel()

        self._model.build()

    def simulate(self, image_bgr: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Simula la percepción prostética de una imagen.

        Parámetros
        ----------
        image_bgr : np.ndarray
            Imagen en BGR (salida del encoder o imagen original).
        target_size : tuple[int, int]
            Tamaño (width, height) al que se redimensiona antes de simular.

        Devuelve
        --------
        np.ndarray
            Imagen de fosfenos simulada, shape (H, W), dtype float32.
        """
        import pulse2percept as p2p

        # Convertir a escala de grises y redimensionar
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, target_size)

        # Crear estímulo
        stimulus = p2p.stimuli.ImageStimulus(gray_resized)

        # Asignar estímulo al implante y simular
        self._implant.stim = stimulus
        percept = self._model.predict_percept(self._implant)

        # Extraer el frame de datos como array 2D
        frame = percept.data[:, :, 0]
        return frame.astype(np.float32)

    def simulate_to_uint8(self, image_bgr: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
        """
        Como simulate(), pero normaliza y devuelve uint8 (0-255) para visualización.
        """
        frame = self.simulate(image_bgr, target_size)
        if frame.max() > frame.min():
            frame = (frame - frame.min()) / (frame.max() - frame.min()) * 255
        return frame.astype(np.uint8)
