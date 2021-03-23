def txt2str(file):
    '''
    Returns strings of sequences from a text file.
    
    Parameters
    ----------
    file : str
        A path to a .txt file containing nucleotide or amino acid sequences.

    Returns
    ----------
    seq: list
        A list of nucleotide sequence strings.
    '''

    seq = []
    with open(file) as file:
        lines = file.readlines()
        last = len(lines) - 1
        for n, line in enumerate(lines):
            if n == last:
                seq.append(line)
            else:
                seq.append(line[:-1])
    return seq


def fa2str(file):
    '''
    Returns strings of sequences from a fasta file.
    
    Parameters
    ----------
    file : str
        A path to a fasta (.fa) file containing 
        nucleotide or amino acid sequences.

    Returns
    ----------
    seq: list
        A list of nucleotide sequence strings.
    '''

    amino = ''
    seq = []
    with open(file) as file:
        f = ''
        lines = file.readlines()
        last = len(lines)
        for n, line in enumerate(lines):
            if n == 0:
                continue
            if n == last:
                f += line
                seq.append(f)
            if line.startswith('>'):
                seq.append(f)
                f = ''
            else:
                f += line[:-1]
    return seq
