import argparse
import logging
import pickle
import re
from collections import defaultdict
from pathlib import Path

import sacrebleu
from tqdm import tqdm
from transformers import AutoTokenizer

from onmt.bin.translate import translate, _get_parser

logger = logging.getLogger(__name__)

def natsorted(iterable):
    return sorted(
        iterable,
        key=lambda f: [
            int(x) if x.isdigit() else x
            for x in re.split("([0-9]+)", str(f)) if x
        ]
    )

def load_references(data_dir, reference_fnames, tokenizer_path_or_name, tokenize_refs=True):
    """
    Load reference sentences and tokenize to match the output
    """
    references = []
    if tokenizer_path_or_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path_or_name)
        
    for reference in reference_fnames:
        with open(Path(data_dir, reference), "r") as infile:
                sents = [l.strip() for l in infile]
                if tokenize_refs:
                    sents = [" ".join(tokenizer.tokenize(l)) for l in sents]
                if not tokenizer_path_or_name:
                    sents = [re.sub("(@@ )|(@@ ?$)", "", l.strip()) for l in sents]
                if "t5" in tokenizer_path_or_name:
                    sents = [l.replace(" ", "").replace("▁", " ") for l in sents]
                if "bert" in tokenizer_path_or_name:
                    sents = [l.replace(" ##", "") for l in sents]
        references.append(sents)

    return references


def eval_checkpoints(checkpoints, references, output_dir, args, pbar=None):
    """
    Evaluate all the checkpoints in a list
    """
    best_bleu = 0.
    input_name = Path(args.graph_fname).stem
    results = []

    if (
        Path(output_dir, f"decoded-{input_name}-{checkpoints[-1].stem}.txt").exists()
        and not args.overwrite
    ):
        if pbar is not None:
            pbar.update(len(checkpoints))
        return

    for i, checkpoint in enumerate(checkpoints):
        checkpoint_name = checkpoint.stem
        output_fpath = Path(output_dir, f"decoded-{input_name}-{checkpoint_name}.txt")

        if args.eval_only or output_fpath.exists() and not args.overwrite:
            with open(output_fpath, "r") as infile:
                decoded = [l.strip() for l in infile]
        else:
            translate_parser = _get_parser()
            opt = translate_parser.parse_args([
                '-model', str(checkpoint),
                '-src', str(Path(args.data_dir, args.nodes_fname)),
                '-graph', str(Path(args.data_dir, args.graph_fname)),
                '-output', str(output_fpath),
                '-beam_size', args.beam_size,
                '-share_vocab',
                '-min_length', args.min_length,
                '-max_length', args.max_length,
                '-length_penalty', 'wu',
                '-alpha', args.alpha,
                '-batch_size', args.batch_size,
                '-gpu', '0'
            ])
            translate(opt)

            # detokenize and save
            with open(output_fpath, "r") as infile:
                if not args.tokenizer_path_or_name: # this is the default
                    decoded = [re.sub("(@@ )|(@@ ?$)", "", l.strip()) for l in infile]
                elif "t5" in args.tokenizer_path_or_name:
                    decoded = [l.replace(" ", "").replace("▁", " ").strip() for l in infile]
                elif "bert" in args.tokenizer_path_or_name:
                    decoded = [l.replace(" ##", "").strip() for l in infile]
            with open(output_fpath, "w") as outfile:
                outfile.write("\n".join(decoded))

        # evaluate
        bleu = sacrebleu.corpus_bleu(decoded, references)
        print(f"{checkpoint_name:12}: {bleu.score:0.2f} BLEU")
        best_bleu = max(best_bleu, bleu.score)
        results.append((checkpoint_name, bleu))

        if pbar is not None:
            pbar.update(1)

    with open(Path(output_dir, f"bleu-{input_name}.txt"), "w") as outfile:
        for checkpoint_name, bleu in results:
            if bleu.score == best_bleu:
                logger.info(f"Best BLEU: {bleu.score:0.2f} @ {checkpoint_name}")
                checkpoint_name = f"{checkpoint_name}*"
            outfile.write(f"{checkpoint_name:12}: {bleu.format()}\n")

    # save the arguments as a best practice
    with open(Path(output_dir, f"args-{input_name}.bin"), "wb") as outfile:
        pickle.dump(args, outfile)
    
    return best_bleu

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fpath", help="File path to saved model(s) (can be a glob)")
    parser.add_argument("--output_dir", help="Where to store outputs")
    parser.add_argument(
        "--no_subdir", default=False, action="store_true",
        help="Do not create a second directory using the `model_fpath` name"
    )
    parser.add_argument("--tokenizer_path_or_name", default="")
    parser.add_argument("--refs_already_tokenized", action="store_true", default=False)

    parser.add_argument("--data_dir", help="Where the data is stored")
    parser.add_argument("--nodes_fname", help="Node filepath")
    parser.add_argument("--graph_fname", help="Graph filepath")
    parser.add_argument("--reference_fnames", nargs="+", help="Reference paths")
    parser.add_argument("--checkpoint_extension", default="pt", help="file extention for a checkpoint")
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--use_agenda_defaults", default=False, action="store_true")
    parser.add_argument("--use_webnlg_defaults", default=False, action="store_true")

    parser.add_argument("--beam_size", default="3")
    parser.add_argument("--alpha", default="3")
    parser.add_argument("--batch_size", default="60")
    parser.add_argument("--min_length", default=None)
    parser.add_argument("--max_length", default=None)

    parser.add_argument("--overwrite", default=False, action="store_true")

    args = parser.parse_args()

    if args.use_webnlg_defaults:
        args.beam_size = "3"
        args.alpha = "3"
        args.min_length = "0" # omnt default
        args.max_length = "100" # omnt default
        args.batch_size = "60"
    if args.use_agenda_defaults:
        args.beam_size = "5"
        args.alpha = "5"
        args.min_length = "0"
        args.max_length = "430"
        args.batch_size = "80"

    # Evaluate BLEU on all models
    # Load the references
    references = load_references(
        data_dir=args.data_dir,
        reference_fnames=args.reference_fnames,
        tokenizer_path_or_name=args.tokenizer_path_or_name,
        tokenize_refs=not args.refs_already_tokenized,
    )

    # Decode the sentences
    paths = list(Path("./").glob(args.model_fpath))
    run_dirs = defaultdict(list)
    for path in paths:
        if path.suffix.endswith(args.checkpoint_extension):
            run_dirs[path.parent].append(path)

    bleus = []
    with tqdm(total = len(paths)) as pbar:
        for dirname, checkpoints  in run_dirs.items():
            checkpoints = natsorted(checkpoints)
            if len(run_dirs) == 1 and args.no_subdir:
                output_dir = Path(args.output_dir)
            else:
                output_dir = Path(args.output_dir, dirname.name)
            output_dir.mkdir(parents=True, exist_ok=True)
            best_bleu = eval_checkpoints(checkpoints, references, output_dir, args, pbar)


