from config import TRAIN_DIAGNOSIS, TRAIN_RED_FLAG, TRAIN_DURATION

def train_model():
    if TRAIN_DIAGNOSIS:
        from model.diagnosis_trainer import train_diagnosis_model
        train_diagnosis_model()

    if TRAIN_RED_FLAG:
        from model.train_red_flag import train_red_flag_model
        train_red_flag_model()

    if TRAIN_DURATION:
        from model.train_duration import train_duration_model
        train_duration_model()


if __name__ == "__main__":
    train_model()   