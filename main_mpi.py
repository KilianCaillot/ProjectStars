import argparse
import os
from src.preprocess_fits import preprocess_fits
from src.extract_objects import extract_objects
from src.extract_spectra import extract_spectrum, save_spectrum
from src.create_database import create_database
from src.utils import clean_directory
from tqdm import tqdm
import time
from mpi4py import MPI
import logging

def setup_logger():
    logger = logging.getLogger('FITS_Processor')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def process_fits_files(input_dir, output_dir, fits_files, logger):
    all_objects = []

    total_objects = len(fits_files)
    pbar = tqdm(total=total_objects, desc="Processing Objects", position=0, leave=True)

    for fits_file in fits_files:
        try:
            file_id = fits_file.split('_cor')[0]
            header, data = preprocess_fits(os.path.join(input_dir, fits_file))
            objects = extract_objects(data)

            output_subdir = os.path.join(output_dir, file_id)
            os.makedirs(output_subdir, exist_ok=True)

            for obj in objects:
                obj['origin'] = file_id
                spectrum = extract_spectrum(data, obj['X'], obj['Y'])
                save_spectrum(spectrum, obj['Object_ID'], output_subdir)
                all_objects.append(obj)

            pbar.update(1)
        except Exception as e:
            logger.error(f"Error processing file {fits_file}: {e}")

    pbar.close()
    return all_objects

def main():
    debut = time.time()
    parser = argparse.ArgumentParser(description='Process FITS files to extract spectra.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing the FITS files.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the extracted spectra.')

    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger = setup_logger()

    if rank == 0:
        clean_directory(args.output_dir)

    comm.Barrier()

    fits_files = [f for f in os.listdir(args.input_dir) if f.endswith('.fits')]
    num_files = len(fits_files)

    chunk_size = num_files // size
    remainder = num_files % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    local_files = fits_files[start:end]

    local_objects = process_fits_files(args.input_dir, args.output_dir, local_files, logger)

    all_objects = comm.gather(local_objects, root=0)

    if rank == 0:
        all_objects = [obj for sublist in all_objects for obj in sublist]
        create_database(all_objects, os.path.join(args.output_dir, '..', 'objects.csv'))

        resultat = time.time() - debut
        logger.info(f"Total execution time: {resultat} seconds")
        logger.info("Without optimization: 1.40 seconds for 70,000 files generated")

if __name__ == "__main__":
    main()
