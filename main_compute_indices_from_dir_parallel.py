__author__ = 'guyot'

#!/usr/bin/env python

"""
    Compute and output acoustic indices from a directory
"""

__author__ = "Patrice Guyot"
__version__ = "0.1"
__credits__ = ["Patrice Guyot"]
__email__ = ["patrice.guyot@mines-ales.fr"]
__status__ = "Development"


from compute_indice import *
from acoustic_index import *
import yaml
from scipy import signal
from csv import writer
import argparse
import os
from multiprocessing import Pool, cpu_count

def process_files(files, data_config, print_statements = False):
    all_results = []
    column_names = None

    for idx_file, filename in enumerate(files):
        # Read signal -------------------------------------
        if print_statements:
            file = AudioFile(filename, verbose=True)
        else:
            file = AudioFile(filename, verbose=False)
           

        # Pre-processing -----------------------------------------------------------------------------------
        if 'Filtering' in data_config:
            if data_config['Filtering']['type'] == 'butterworth':
                if print_statements  or idx_file == 0:
                    print('- Pre-processing - High-Pass Filtering:', data_config['Filtering'])
                freq_filter = data_config['Filtering']['frequency']
                Wn = freq_filter/float(file.niquist)
                order = data_config['Filtering']['order']
                [b,a] = signal.butter(order, Wn, btype='highpass')
                # to plot the frequency response
                #w, h = signal.freqz(b, a, worN=2000)
                #plt.plot((file.sr * 0.5 / np.pi) * w, abs(h))
                #plt.show()
                file.process_filtering(signal.filtfilt(b, a, file.sig_float))
            elif data_config['Filtering']['type'] == 'windowed_sinc':
                if print_statements or idx_file == 0:
                    print('- Pre-processing - High-Pass Filtering:', data_config['Filtering'])
                freq_filter = data_config['Filtering']['frequency']
                fc = freq_filter / float(file.sr)
                roll_off = data_config['Filtering']['roll_off']
                b = roll_off / float(file.sr)
                N = int(np.ceil((4 / b)))
                if not N % 2: N += 1  # Make sure that N is odd.
                n = np.arange(N)
                # Compute a low-pass filter.
                h = np.sinc(2 * fc * (n - (N - 1) / 2.))
                w = np.blackman(N)
                h = h * w
                h = h / np.sum(h)
                # Create a high-pass filter from the low-pass filter through spectral inversion.
                h = -h
                h[(N - 1) / 2] += 1
                file.process_filtering(np.convolve(file.sig_float, h))



        # Compute Indices -----------------------------------------------------------------------------------
        if print_statements or idx_file == 0:
            print('- Compute Indices')
        ci = data_config['Indices'] # use to simplify the notation
        for index_name in ci:  # iterate over the index names (key of dictionary in the yml file)


            if index_name == 'Acoustic_Complexity_Index':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, _ = compute_spectrogram(file, **ci[index_name]['spectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                j_bin = int(ci[index_name]['arguments']['j_bin'] * file.sr / ci[index_name]['spectro']['windowHop']) # transform j_bin in samples
                main_value, temporal_values = methodToCall(spectro, j_bin)
                file.indices[index_name] = Index(index_name, temporal_values=temporal_values, main_value=main_value)
                

            elif index_name == 'Acoustic_Diversity_Index':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                freq_band_Hz = ci[index_name]['arguments']['max_freq'] / ci[index_name]['arguments']['freq_step']
                windowLength = int(file.sr / freq_band_Hz)
                spectro,_ = compute_spectrogram(file, windowLength=windowLength, windowHop= windowLength, scale_audio=True, square=False, windowType='hamming', centered=False, normalized= False )
                main_value = methodToCall(spectro, freq_band_Hz, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Acoustic_Evenness_Index':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                freq_band_Hz = ci[index_name]['arguments']['max_freq'] / ci[index_name]['arguments']['freq_step']
                windowLength = int(file.sr / freq_band_Hz)
                spectro,_ = compute_spectrogram(file, windowLength=windowLength, windowHop= windowLength, scale_audio=True, square=False, windowType='hamming', centered=False, normalized= False )
                main_value = methodToCall(spectro, freq_band_Hz, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Bio_acoustic_Index':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, frequencies = compute_spectrogram(file, **ci[index_name]['spectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(spectro, frequencies, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Normalized_Difference_Sound_Index':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(file, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'RMS_energy':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                temporal_values = methodToCall(file, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, temporal_values=temporal_values)


            elif index_name == 'Spectral_centroid':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, frequencies = compute_spectrogram(file, **ci[index_name]['spectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                temporal_values = methodToCall(spectro, frequencies)
                file.indices[index_name] = Index(index_name, temporal_values=temporal_values)


            elif index_name == 'Spectral_Entropy':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, _ = compute_spectrogram(file, **ci[index_name]['spectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(spectro)
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Temporal_Entropy':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(file, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'ZCR':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                temporal_values = methodToCall(file, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, temporal_values=temporal_values)


            elif index_name == 'Wave_SNR':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                values = methodToCall(file, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, values=values)


            elif index_name == 'NB_peaks':
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, frequencies = compute_spectrogram(file, **ci[index_name]['spectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(spectro, frequencies, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Acoustic_Diversity_Index_NR': # Acoustic_Diversity_Index with Noise Removed spectrograms
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                freq_band_Hz = ci[index_name]['arguments']['max_freq'] / ci[index_name]['arguments']['freq_step']
                windowLength = int(file.sr / freq_band_Hz)
                spectro,_ = compute_spectrogram(file, windowLength=windowLength, windowHop= windowLength, scale_audio=True, square=False, windowType='hamming', centered=False, normalized= False )
                spectro_noise_removed = remove_noiseInSpectro(spectro, **ci[index_name]['remove_noiseInSpectro'])
                main_value = methodToCall(spectro_noise_removed, freq_band_Hz, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Acoustic_Evenness_Index_NR': # Acoustic_Evenness_Index with Noise Removed spectrograms
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                methodToCall = globals().get(ci[index_name]['function'])
                freq_band_Hz = ci[index_name]['arguments']['max_freq'] / ci[index_name]['arguments']['freq_step']
                windowLength = int(file.sr / freq_band_Hz)
                spectro,_ = compute_spectrogram(file, windowLength=windowLength, windowHop= windowLength, scale_audio=True, square=False, windowType='hamming', centered=False, normalized= False )
                spectro_noise_removed = remove_noiseInSpectro(spectro, **ci[index_name]['remove_noiseInSpectro'])
                main_value = methodToCall(spectro_noise_removed, freq_band_Hz, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Bio_acoustic_Index_NR': # Bio_acoustic_Index with Noise Removed spectrograms
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, frequencies = compute_spectrogram(file, **ci[index_name]['spectro'])
                spectro_noise_removed = remove_noiseInSpectro(spectro, **ci[index_name]['remove_noiseInSpectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(spectro_noise_removed, frequencies, **ci[index_name]['arguments'])
                file.indices[index_name] = Index(index_name, main_value=main_value)


            elif index_name == 'Spectral_Entropy_NR': # Spectral_Entropy with Noise Removed spectrograms
                if print_statements or idx_file == 0:
                    print('\tCompute', index_name)
                spectro, _ = compute_spectrogram(file, **ci[index_name]['spectro'])
                spectro_noise_removed = remove_noiseInSpectro(spectro, **ci[index_name]['remove_noiseInSpectro'])
                methodToCall = globals().get(ci[index_name]['function'])
                main_value = methodToCall(spectro_noise_removed)
                file.indices[index_name] = Index(index_name, main_value=main_value)

        # Extract column names from the first processed file
        if idx_file == 0:
            keys = ['filename']
            for idx, current_index in file.indices.items():
                for key in current_index.__dict__:
                    if key != 'name':
                        keys.append(f"{idx}__{key}")
            column_names = keys  # Save column names

        # Append processed data for CSV writing
        values = [file.file_name]
        for idx, current_index in file.indices.items():
            for key in current_index.__dict__:
                if key != 'name':
                    values.append(getattr(current_index, key))
        all_results.append(values)

    return column_names, all_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help='yaml config file', nargs='?', const='yaml/config_014_butter.yaml', default='yaml/config_014_butter.yaml', type=str)
    parser.add_argument("audio_dir", help='audio directory', nargs='?', const='audio_files', default='audio_files', type=str)
    parser.add_argument("output_csv_file", help='output csv file', nargs='?', const='dict_all.csv', default='dict_all.csv', type=str)
    parser.add_argument("--processes", help="number of processes to use", type=int, default=cpu_count() - 1)
    parser.add_argument("--print_statements", help="whether to print which index is being computed", type=bool, default=False)

    args = parser.parse_args()

    # Load config file
    with open(args.config_file, 'r') as stream:
        data_config = yaml.load(stream, Loader=yaml.FullLoader)

    # Get list of audio files
    all_audio_file_path = [os.path.join(path, name)
                           for path, subdirs, files in os.walk(args.audio_dir)
                           for name in files if name.endswith(".wav") and not name.startswith(".")]
    print("-", len(all_audio_file_path), "files found in the directory", args.audio_dir, ":\n")

    # Split audio files for multiprocessing
    n_cpus = args.processes
    file_paths_split = np.array_split(all_audio_file_path, n_cpus)

    # Run processing on multiple cores
    with Pool(n_cpus) as pool:
        results = pool.starmap(process_files, [(paths, data_config) for paths in file_paths_split])

    # Extract column names and results
    column_names = results[0][0]  # Use the first core's column names for CSV header
    all_results = [result for _, result_list in results for result in result_list]

    # Write results to CSV
    with open(args.output_csv_file, 'w', newline='') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(column_names)  # Write header
        writer_object.writerows(all_results)  # Write data

if __name__ == '__main__':
    main()
