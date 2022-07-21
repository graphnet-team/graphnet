def frame_is_montecarlo(frame):
    return ("MCInIcePrimary" in frame) or ("I3MCTree" in frame)


def frame_is_noise(frame):
    try:
        frame["I3MCTree"][0].energy
        return False
    except:  # noqa: E722
        try:
            frame["MCInIcePrimary"].energy
            return False
        except:  # noqa: E722
            return True
