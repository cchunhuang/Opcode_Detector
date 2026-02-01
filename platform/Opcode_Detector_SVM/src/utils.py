import os
import r2pipe
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

# Get logger that will inherit configuration from parent module
logger = logging.getLogger(__name__)

def Extraction(filename):
    """
    Extract opcode sequence from a binary file using radare2.
    
    Args:
        filename: Path to the binary file
        
    Returns:
        list: Opcode sequence, or empty list if extraction fails
    """
    # Check file existence
    if not os.path.exists(filename):
        filename = os.path.join(os.path.dirname(filename), os.path.basename(filename)[:2], os.path.basename(filename))
    if not os.path.exists(filename):
        logger.error(f"File not found: {filename}")
        raise FileNotFoundError(f"File not found: {filename}")
    
    try:
        # OpenFile
        r2 = r2pipe.open(filename)
        # Analyze  
        r2.cmd('aaaa')
        # Extraction
        OpcodeSequence = []
        DisassembleJ = r2.cmdj('pdj $s')
        
        if DisassembleJ:
            for instruction in DisassembleJ:
                try:
                    if instruction.get('opcode'):
                        opcode = instruction['opcode'].split(' ')[0]
                        if opcode and opcode != "invalid":
                            OpcodeSequence.append(opcode)
                except (KeyError, IndexError, AttributeError):
                    pass
        
        r2.quit()
        
        if not OpcodeSequence:
            logger.warning(f"No opcodes extracted from {filename}")
        
        return OpcodeSequence
        
    except Exception as e:
        logger.error(f"Error extracting opcodes from {filename}: {str(e)}")
        return []

def Extraction_batch(filenames, dataset_folder, n_jobs=None):
    """
    Extract opcode sequences from multiple binary files in parallel.
    
    Args:
        filenames: List of filenames to process
        dataset_folder: Base folder containing the dataset files
        n_jobs: Number of parallel processes to use. If None, uses all available CPUs.
                If -1, uses all available CPUs. If 1, runs sequentially.
        
    Returns:
        list: List of opcode sequences for each file
    """
    # Determine number of jobs
    if n_jobs is None:
        n_jobs = cpu_count() - 2
    elif n_jobs <= -1:
        n_jobs = cpu_count() + n_jobs
    elif n_jobs < 1:
        n_jobs = 1
    
    # Construct full file paths
    file_paths = [os.path.join(dataset_folder, f) for f in filenames]
    
    # Use sequential processing if n_jobs is 1
    if n_jobs == 1:
        logger.info("Using sequential processing (n_jobs=1)")
        return [Extraction(fp) for fp in file_paths]
    
    # Use parallel processing
    logger.info(f"Using parallel processing with {n_jobs} processes")
    try:
        with Pool(processes=n_jobs) as pool:
            results = pool.map(Extraction, file_paths)
        return results
    except Exception as e:
        logger.error(f"Error in parallel extraction: {str(e)}")
        logger.info("Falling back to sequential processing")
        return [Extraction(fp) for fp in file_paths]

def Vectorize(sequence, top_features_path="./top_features_1.npy"):
    """
    Vectorize an opcode sequence using n-gram features.
    
    Args:
        sequence: List of opcodes
        top_features_path: Path to the top features file
        
    Returns:
        numpy.ndarray: Vectorized representation
    """
    # Handle empty or very short sequences
    if not sequence or len(sequence) < 2:
        logger.warning(f"Sequence too short or empty (length: {len(sequence)}), returning zero vector")
        top_feature = np.load(top_features_path)
        return np.zeros((1, len(top_feature)))
    
    seq = ' '.join(str(i) for i in sequence)
    X = [seq]
    
    try:
        vectorizer_count = CountVectorizer(ngram_range=(2, 4))
        x_count = vectorizer_count.fit_transform(X)
        
        feature_list = vectorizer_count.get_feature_names_out().tolist()
        
        top_feature = np.load(top_features_path)
        tmp = []
        
        for j in range(len(top_feature)):
            if top_feature[j] in feature_list:
                tmp.append(x_count[:, feature_list.index(top_feature[j])][0, 0])
            else:
                tmp.append(0)
        
        tmp = np.asarray(tmp).reshape(1, -1)
        return tmp
        
    except ValueError as e:
        # If CountVectorizer fails (e.g., sequence still too short for n-grams)
        logger.warning(f"Vectorization failed: {str(e)}, returning zero vector")
        top_feature = np.load(top_features_path)
        return np.zeros((1, len(top_feature)))
