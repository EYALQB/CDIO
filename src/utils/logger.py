import logging
import os

def get_logger(name: str = "project"):
    """
    Configura i retorna un objecte de registre (logger) per al projecte.

    El logger mostra missatges per pantalla (nivell INFO)
    i també desa tots els detalls a un fitxer 'logs/project.log' (nivell DEBUG).

    Paràmetres
    ----------
    name : str
        Nom identificador del logger (per defecte: 'project').

    Retorna
    -------
    logging.Logger
        Objecte configurat per registrar missatges.
    """

    # --- 1️⃣ Crear carpeta de logs si no existeix
    os.makedirs("logs", exist_ok=True)

    # --- 2️⃣ Crear i configurar el logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # --- 3️⃣ Evitar duplicar gestors si el logger ja està creat
    if not logger.handlers:
        # Sortida per pantalla (nivell informatiu)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Sortida a fitxer (nivell complet)
        fh = logging.FileHandler("logs/project.log", mode="a")
        fh.setLevel(logging.DEBUG)

        # Format del missatge
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)

        # Afegim gestors al logger
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
