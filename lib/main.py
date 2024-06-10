import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'assets')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_youtube_video_id(url, ignore_playlist=True):
    """
    Extracts the video ID from a YouTube URL.
    """
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]
        if query.path[:8] == '/shorts/':
            return query.path.split('/')[2]

    return None



def yt_download(link):
    """
    Downloads the best audio format from a YouTube link.
    """
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path


def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)


def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'No model file exists in {model_dir}.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''


def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None
    backup_vocals_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

        elif file.endswith('_Vocals_Backup.wav'):
            backup_vocals_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path, backup_vocals_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)


def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Downloading song...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    orig_song_path = convert_to_stereo(orig_song_path)

    display_progress('[~] Separating Vocals from Instrumental...', 0.1, is_webui, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Kim_Vocal_2.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)

    display_progress('[~] Separating Main Vocals from Backup Vocals...', 0.2, is_webui, progress)
    backup_vocals_path, main_vocals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'UVR_MDXNET_KARA_2.onnx'), vocals_path, suffix='Backup', invert_suffix='Main', denoise=True)

    display_progress('[~] Applying DeReverb to Vocals...', 0.3, is_webui, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), main_vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path


def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()


def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
        ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def merge_audios(audio_paths, output_path):
    combined = AudioSegment.from_file(audio_paths[0])
    for path in audio_paths[1:]:
        combined = combined.overlay(AudioSegment.from_file(path))
    combined.export(output_path, format='wav')


def process_and_save_song(song_input, input_type, voice_model, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, progress, is_webui=False):
    song_id = get_hash(song_input)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    song_output_dir = os.path.join(output_dir, song_id)
    if not os.path.exists(song_output_dir):
        os.makedirs(song_output_dir)

    if input_type == 'yt' and not get_youtube_video_id(song_input):
        raise_exception('[!] Invalid YouTube link.', is_webui)

    mdx_model_params = {
        'demucs_model_path': os.path.join(mdxnet_models_dir, 'models_demucs.h5'),
        'mdx_model_path': os.path.join(mdxnet_models_dir, 'models_mdx.h5'),
        'output_path': output_dir,
        'noise_protect': 0.33,
        'voc_model_path': os.path.join(mdxnet_models_dir, 'models_vocal.h5')
    }

    try:
        orig_song_path, vocals_path, instrumentals_path, main_vocals_path, backup_vocals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)

        display_progress('[~] Changing Main Vocals to Target Voice...', 0.4, is_webui, progress)
        pitch_shifted_main_vocals_path = pitch_shift(main_vocals_dereverb_path, pitch_change)
        output_vocals_path = os.path.join(song_output_dir, 'main_vocals_changed.wav')
        voice_change(voice_model, pitch_shifted_main_vocals_path, output_vocals_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Adding Audio Effects...', 0.5, is_webui, progress)
        final_output_vocals_path = add_audio_effects(output_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)

        display_progress('[~] Merging Vocal and Instrumental Tracks...', 0.6, is_webui, progress)
        final_output_path = os.path.join(output_dir, f'{os.path.basename(orig_song_path)}_{voice_model}_vocal_conversion.wav')
        merge_audios([final_output_vocals_path, instrumentals_path], final_output_path)

        display_progress('[~] Done!', 1.0, is_webui, progress)
    except Exception as e:
        raise_exception(f'[!] Processing failed: {str(e)}', is_webui)
    finally:
        with suppress(FileNotFoundError):
            os.remove(orig_song_path)

    return final_output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process song with RVC.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input song file or YouTube link.')
    parser.add_argument('--type', type=str, required=True, choices=['local', 'yt'], help='Type of input: "local" for a file, "yt" for a YouTube link.')
    parser.add_argument('--voice_model', type=str, required=True, help='Name of the voice model to use.')
    parser.add_argument('--pitch_change', type=float, required=False, default=0, help='Pitch change amount in semitones.')
    parser.add_argument('--f0_method', type=str, required=False, default='crepe', help='F0 method to use.')
    parser.add_argument('--index_rate', type=float, required=False, default=1.0, help='Index rate.')
    parser.add_argument('--filter_radius', type=float, required=False, default=3.0, help='Filter radius.')
    parser.add_argument('--rms_mix_rate', type=float, required=False, default=0.25, help='RMS mix rate.')
    parser.add_argument('--protect', type=float, required=False, default=0.33, help='Protection rate.')
    parser.add_argument('--crepe_hop_length', type=int, required=False, default=128, help='Crepe hop length.')
    parser.add_argument('--reverb_rm_size', type=float, required=False, default=0.3, help='Reverb room size.')
    parser.add_argument('--reverb_wet', type=float, required=False, default=0.25, help='Reverb wet level.')
    parser.add_argument('--reverb_dry', type=float, required=False, default=0.75, help='Reverb dry level.')
    parser.add_argument('--reverb_damping', type=float, required=False, default=0.5, help='Reverb damping.')

    args = parser.parse_args()

    process_and_save_song(
        song_input=args.input,
        input_type=args.type,
        voice_model=args.voice_model,
        pitch_change=args.pitch_change,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        filter_radius=args.filter_radius,
        rms_mix_rate=args.rms_mix_rate,
        protect=args.protect,
        crepe_hop_length=args.crepe_hop_length,
        reverb_rm_size=args.reverb_rm_size,
        reverb_wet=args.reverb_wet,
        reverb_dry=args.reverb_dry,
        reverb_damping=args.reverb_damping,
        progress=None
    )
