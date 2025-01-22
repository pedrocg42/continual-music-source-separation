import os

import streamlit as st

from analysis.separator import Separator
from analysis.utils.audio import plot_audio, read_audio, write_audio

available_models = {
    "Demucs Vocals": {
        "path": "./../output/demucs-adam_vocals/model.pth.tar",
        "sources": ["Vocals 👄"],
    },
    "Demucs Drums": {
        "path": "./../output/demucs-adam_drums/model.pth.tar",
        "sources": ["Drums 🥁"],
    },
    "Demucs Bass": {
        "path": "./../output/demucs-adam_bass/model.pth.tar",
        "sources": ["Bass 🎸"],
    },
    "Demucs Other": {
        "path": "./../output/demucs-adam_other/model.pth.tar",
        "sources": ["Other 🎻"],
    },
}


@st.cache_resource(show_spinner=True)
def load_separator(model_name: str) -> Separator:
    model_dict = available_models[model_name]
    separator = Separator(
        model_path=model_dict["path"], sources_names=model_dict["sources"], **model_dict.get("kwargs", {})
    )
    return separator


def audio_column(col):
    with col:
        st.header("Input")
        with st.form("Prepare Input"):
            st.session_state["uploaded_file"] = st.file_uploader("Select your song 🎵", type=["wav", "mp3"])
            sample_rate = st.number_input("Sample Rate", min_value=1000, max_value=44100, value=44100)

            read_data = st.form_submit_button("Read Data")

        if read_data:
            if not st.session_state["uploaded_file"]:
                st.warning("Please upload a valid audio", icon="🚨")
                st.stop()

            st.session_state["audio"], st.session_state["sample_rate"] = read_audio(
                st.session_state["uploaded_file"], sample_rate=sample_rate
            )

        if st.session_state["audio"] is not None:
            st.success(icon="🚀", body=f"Audio correcly loaded with shape {st.session_state['audio'].shape}")
            plot_audio(st.session_state["audio"])
            st.audio(st.session_state["audio"], sample_rate=st.session_state["sample_rate"])


def models_column(col: st.container):
    with col:
        st.header("Models")
        with st.form("Select Models"):
            models_selected = st.multiselect("Select your models", options=available_models.keys())
            load_models = st.form_submit_button("Load Models")

        if load_models:
            st.subheader("Loaded Models")
            for model_name in models_selected:
                separator = load_separator(model_name)
                st.session_state["separators"][model_name] = separator

        for model_name, separator in st.session_state["separators"].items():
            st.success(icon="🚀", body=f"Model loaded {model_name} | sources: {separator.sources_names}")

    should_separate = st.button("Separate Sources 🚀")

    if should_separate:
        if len(st.session_state["separators"]) == 0 or st.session_state["audio"] is None:
            st.warning("Please make sure to load valid audio and models", icon="🚨")
            st.stop()
    else:
        st.stop()


def result_section():
    st.header("Results")
    for separator_name, separator in st.session_state["separators"].items():
        st.subheader(separator_name)
        sources = separator(st.session_state["audio"].copy())
        for source, source_name in zip(sources, separator.sources_names, strict=True):
            st.text(source_name)
            plot_audio(source)
            st.audio(source, sample_rate=st.session_state["sample_rate"])
            audio_file = f"{os.path.splitext(st.session_state['uploaded_file'].name)[0]} - {source_name}.wav"
            write_audio(audio_file, audio=source, sample_rate=st.session_state["sample_rate"])
            st.success(f"Audio saved in {audio_file}")


def source_separation():
    st.set_page_config(layout="wide", page_icon="🎶", page_title="Source Separation")

    # Initialization
    if "started" not in st.session_state:
        st.session_state["started"] = True
        st.session_state["uploaded_file"] = None
        st.session_state["audio"] = None
        st.session_state["sample_rate"] = None
        st.session_state["separators"] = {}

    st.title("Music Source Separation")

    col1, col2 = st.columns(2)
    audio_column(col1)
    models_column(col2)
    result_section()


if __name__ == "__main__":
    source_separation()
