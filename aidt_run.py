import argparse
import json
import numpy as np
import os
import pyrosetta
import torch
import run
import math
import pathlib

def main(args):

    pyrosetta.init('-mute all')

    # -------------------------------------------------------------------------
    # parse all arguments
    # -------------------------------------------------------------------------

    if args.seed:
        seed = args.seed
        print("Passed seed", seed)
    else:
        seed = int(np.random.randint(0, high=99999, size=1, dtype=int)[0])
        print("Random seed", seed)

    input_directory = args.in_path
    if input_directory[-1] != '/':
        input_directory += '/'

    output_directory = args.out_path
    if output_directory[-1] != '/':
        output_directory += '/'
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    design_chain = args.chains_to_design

    batch_size = args.batch_size
    designs_per_input = args.designs_per_input

    surface_bias_value = args.surface_bias
    boundary_bias_value = args.boundary_bias
    
    # -------------------------------------------------------------------------
    # collect all pdb files
    # -------------------------------------------------------------------------

    pdb_files = []
    with os.scandir(input_directory) as entries:
        for entry in entries:
            if entry.is_file and entry.name[-4:] == '.pdb':
                pdb_files.append(entry.path)
    
    print("Found pdb files:")
    print(pdb_files)

    # -------------------------------------------------------------------------
    # prepare pdb file paths for ligand mpnn
    # -------------------------------------------------------------------------

    pdb_files_dict = {}
    for path in pdb_files:
        pdb_files_dict[path] = ''
    pdb_path_multi = output_directory + 'pdb_files.json'

    with open(pdb_path_multi, 'w') as file:
        json.dump(pdb_files_dict, file, indent=1)

    print("Created MPNN input json")

    # -------------------------------------------------------------------------
    # parse interface file
    # -------------------------------------------------------------------------

    interface_res = []
    interface_file = input_directory + 'interface.txt'
    with open(interface_file) as file:
        interface_res = file.readline().strip().split()
    interface_indices = torch.tensor([ int(res[1:]) for res in interface_res ])
    
    print("Desing only chain:", design_chain)
    print("Defined interface positions ommited from design:")
    print(interface_res)

    # -------------------------------------------------------------------------
    # compute rosetta layers of residues for all input pdb files
    # -------------------------------------------------------------------------
    
    core_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    core_sel.set_layers(True, False, False)
    boundary_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    boundary_sel.set_layers(False, True, False)
    surface_sel = pyrosetta.rosetta.core.select.residue_selector.LayerSelector()
    surface_sel.set_layers(False, False, True)

    print("Loading pdb files into Rosetta for residue layer determination...")
    residue_layers = []
    chain = []
    for pdb in pdb_files:
        pose = pyrosetta.pose_from_file(pdb)

        core_mask = core_sel.apply(pose)
        core_mask = torch.tensor(list(core_mask), dtype=torch.int)
        boundary_mask = boundary_sel.apply(pose)
        boundary_mask = torch.tensor(list(boundary_mask), dtype=torch.int) * 2
        surface_mask = surface_sel.apply(pose)
        surface_mask = torch.tensor(list(surface_mask), dtype=torch.int) * 3
        residue_layers.append(core_mask + boundary_mask + surface_mask)

        if len(chain) == 0:
            pdb_info = pose.pdb_info()
            for i in range(1, pose.total_residue()+1):
                chain.append(pdb_info.chain(i))
        
    chain = np.array(chain)
    chain_mask = torch.tensor(chain == design_chain)

    chain_pos = torch.cumsum(chain_mask.int(), dim=0)
    interface_mask = torch.isin(chain_pos, interface_indices)

    residue_layers = torch.stack(residue_layers)
    # single dimension tensor with the same length as the amino acid chains, 0:ignore, 1: core, 2: boundary, 3: surface, 4: interface
    residue_layers = torch.max(residue_layers, dim=0).values
    residue_layers[~chain_mask] = 0
    residue_layers[interface_mask] = 4

    layer_counts = torch.bincount(residue_layers)
    layer_names = ['ignore', 'core', 'boundary', 'surface', 'interface']
    print("Found residue layers:")
    for i in range(len(layer_names)):
        print('\t', layer_names[i].ljust(10), layer_counts[i].item())

    # -------------------------------------------------------------------------
    # add biases
    # -------------------------------------------------------------------------

    biases = {}
    surface_bias = {"R": surface_bias_value, "H": surface_bias_value, "K": surface_bias_value}
    for pos in chain_pos[residue_layers == 3]:
        key = design_chain + str(pos.item())
        biases[key] = surface_bias
    boundary_bias = {"R": boundary_bias_value, "H": boundary_bias_value, "K": boundary_bias_value}
    for pos in chain_pos[residue_layers == 2]:
        key = design_chain + str(pos.item())
        biases[key] = boundary_bias

    bias_file = output_directory + 'bias.json'
    with open(bias_file, 'w') as file:
        json.dump(biases, file, indent=1)

    print("Created bias file with", len(biases), "biases. Surface:", surface_bias_value, "Boundary:", boundary_bias_value)

    # -------------------------------------------------------------------------
    # run mpnn
    # -------------------------------------------------------------------------

    total_batches = math.ceil(designs_per_input / batch_size)

    print("Starting SolubleMPNN with", total_batches, "batches.")
    print("\tBatch size:", batch_size)
    print("\tDesired total designs:", designs_per_input)
    print("\tActual designs:", total_batches*batch_size)
    print(len(pdb_files), "pdbs are used as input. Total designs:", total_batches*batch_size*len(pdb_files))

    mpnn_argparser = run.define_argparser()
    mpnn_args = mpnn_argparser.parse_args([
        '--model_type', "soluble_mpnn",
        '--seed', str(seed),
        '--pdb_path_multi', pdb_path_multi,
        '--out_folder', output_directory,
        '--save_stats', str(1),
        '--fixed_residues', " ".join(interface_res),
        '--batch_size', str(batch_size),
        '--number_of_batches', str(total_batches),
        '--chains_to_design', design_chain,
        '--bias_AA_per_residue', bias_file,
        '--verbose', '0',
    ])

    run.main(mpnn_args)

    print("Done designing.")

    # -------------------------------------------------------------------------
    # Collect and combine results
    # -------------------------------------------------------------------------

    pt_files = list(pathlib.Path(output_directory + '/stats').glob('*.pt'))
    data_list = [torch.load(f) for f in pt_files]
    print("Loaded statistics from", len(data_list),"input pdbs:")
    for key in data_list[0]:
        try:
            print('\t', key, data_list[0][key].size())
        except:
            print('\t', key, data_list[0][key])
    print('Totalling', len(data_list)*data_list[0]['generated_sequences'].size()[0], 'sequences')

    # because Torben is doing so, we will now average all probs
    # This is to mitigate sampling effects, because the probs at each position change depending on which position was already fixed before
    # Additionally we combine all relaxed structures into the average to incorporate some flexibility
    all_probs = [data['sampling_probs'] for data in data_list]
    all_probs = torch.cat(all_probs, dim=0)
    print('Combined all probabilities into a tensor of size', all_probs.size())

    mean_probs = torch.mean(all_probs, dim=0)
    print('And averaged into', mean_probs.size())

    AA = np.array(list('ACDEFGHIKLMNPQRSTVWYX'))
    wt_aa = data_list[0]['native_sequence']
    max_probs, max_prob_aa = torch.max(mean_probs, dim=1)
    static_aa = wt_aa.clone()
    static_aa[torch.logical_or(residue_layers == 1, torch.logical_or(residue_layers == 2, residue_layers == 3))] = 0
    consensus_aa = max_prob_aa + static_aa
    
    pos_res = ['R', 'H', 'K']
    neg_res = ['D', 'E']

    binder_wt_seq = AA[wt_aa][residue_layers != 0]
    binder_design_seq = AA[consensus_aa][residue_layers != 0]
    changed_pos = []
    charges_design = []
    charges_wt = []
    pos_wt = 0
    neg_wt = 0
    pos_des = 0
    neg_des = 0
    for i in range(len(binder_wt_seq)):
        if binder_wt_seq[i] == binder_design_seq[i]:
            changed_pos.append('_')
        else:
            changed_pos.append('X')

        if binder_design_seq[i] in pos_res:
            charges_design.append(1)
            pos_des += 1
        elif binder_design_seq[i] in neg_res:
            charges_design.append(-1)
            neg_des += 1
        else:
            charges_design.append(0)

        if binder_wt_seq[i] in pos_res:
            charges_wt.append(1)
            pos_wt += 1
        elif binder_wt_seq[i] in neg_res:
            charges_wt.append(-1)
            neg_wt += 1
        else:
            charges_wt.append(0)

    charges_symbols = np.array(list('- +'))
    charges_design = np.array(charges_design)
    charges_wt = np.array(charges_wt)
    charges_delta = charges_design - charges_wt

    positive_delta = []
    negative_delta = []
    for d in charges_delta:
        if d > 0:
            positive_delta.append(" xX"[d])
            negative_delta.append(" ")
        if d < 0:
            positive_delta.append(" ")
            negative_delta.append(" xX"[d*-1])
        if d == 0:
            positive_delta.append(" ")
            negative_delta.append(" ")

    print("\nReport:")
    print("Designed sequence:", "".join(binder_design_seq))
    print("Wildtype sequence:", "".join(binder_wt_seq))
    print("Changed positions:", "".join(changed_pos))
    print("Layers           :", np.array2string(residue_layers[residue_layers != 0].numpy(), separator='', max_line_width=9999, prefix='', suffix='')[1:-1])
    print()
    print("Designed charges :", "".join(charges_symbols[charges_design + 1]))
    print("Wildtype charges :", "".join(charges_symbols[charges_wt + 1]))
    print("Charge delta neg :", "".join(negative_delta))
    print("Charge delta pos :", "".join(positive_delta))
    print()
    print("                 Wildtype Designed Delta")
    print("Negative charges", str(neg_wt).rjust(8), str(neg_des).rjust(8), str(neg_des-neg_wt).rjust(5))
    print("Positive charges", str(pos_wt).rjust(8), str(pos_des).rjust(8), str(pos_des-pos_wt).rjust(5))

    seq_file = output_directory + 'results.fa'
    if not os.path.exists(seq_file):
        with open(seq_file, 'w') as file:
            file.write(">Native sequence\n")
            file.write("".join(binder_wt_seq) + '\n')
            print("Created", seq_file, 'with wildtype sequence')

    print("Appending new sequence to", seq_file)
    with open(seq_file, 'a') as file:
        file.write(">Design " + str(seed) + "\n")
        file.write("".join(binder_design_seq) + '\n')

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog='AI-DT LigandMPNN pipeline',
        description='This is a wrapper to automatically process input files and create Torben style LigandMPNN call.',
        epilog='Proprietary software from AI-Driven Therapeutics GmbH. Usage outside of the company will be fined with 1 Million Euros. Loser.'
    )

    argparser.add_argument(
        "--batch_size",
        type=int,
        help="Set batch size for GPU processing.",
        default=20,
        metavar='20'
    )

    argparser.add_argument(
        "--designs_per_input",
        type=int,
        help="Set how many new sequences should be designed per input pdb",
        default=100,
        metavar='100'
    )

    argparser.add_argument(
        "--surface_bias",
        type=float,
        help="Set negative bias to downweight positively charged surface residues.",
        default=-5.0,
        metavar='-5.0'
    )

    argparser.add_argument(
        "--boundary_bias",
        type=float,
        help="Set negative bias to downweight positively charged boundary residues.",
        default=-2.5,
        metavar='-2.5'
    )

    argparser.add_argument(
        "--seed",
        type=int,
        help="Set seed for torch, numpy, and python random.",
        metavar='42069'
    )

    argparser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to the input directory. Expecting at least one pdb file and exactly one interface.txt file. Multiple pdbs are expected to have the same sequence.",
        metavar='path/to/input_dir/'
    )

    argparser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the output directory. Files with the same name will be overwritten",
        metavar='path/to/output_dir/'
    )

    argparser.add_argument(
        "--chains_to_design",
        type=str,
        required=True,
        help="Specify which chains to redesign, all others will be kept fixed, 'A,B,C,F'",
    )

    # TODO: Remove default options
    args = argparser.parse_args([
        "--in_path", "test",
        "--out_path", "test_out",
        "--chains_to_design", "C",
        #"--designs_per_input", "5"
    ])
    main(args)