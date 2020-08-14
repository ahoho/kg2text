import argparse
from pathlib import Path
import re

import sacrebleu

from onmt.bin.translate import translate, _get_parser

from transformers import AutoTokenizer

def natsorted(iterable):
    return sorted(
        iterable,
        key=lambda f: [
            int(x) if x.isdigit() else x
            for x in re.split("([0-9]+)", str(f)) if x
        ]
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_fpath", help="File path to saved model(s) (can be a glob)")
    parser.add_argument("--output_dir", help="Where to store outputs")
    parser.add_argument("--tokenizer_path_or_name", default="")

    parser.add_argument("--data_dir", help="Where the data is stored")
    parser.add_argument("--nodes_fname", help="Node filepath")
    parser.add_argument("--graph_fname", help="Graph filepath")
    parser.add_argument("--reference_fnames", nargs="+", help="Reference paths")
    parser.add_argument("--eval_only", action="store_true", default=False)

    parser.add_argument("--beam_size", default="3")

    args = parser.parse_args()

    # Evaluate BLEU on all models
    # Load the references
    references = []
    for reference in args.reference_fnames:
        with open(Path(args.data_dir, reference), "r") as infile:
            if not args.tokenizer_path_or_name: # this is the default
                sents = [l.strip() for l in infile]
            else:
                tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path_or_name)
                sents = [" ".join(tokenizer.tokenize(l)) for l in infile]
                if "t5" in args.tokenizer_path_or_name:
                    sents = [l.replace(" ", "").replace("▁", " ") for l in sents]
                if "bert" in args.tokenizer_path_or_name:
                    sents = [l.replace(" ##", "") for l in sents]
        references.append(sents)

    # Decode the sentences
    input_name = Path(args.graph_fname).stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_fpaths = natsorted(Path("./").glob(args.model_fpath))
    best_bleu = 0.

    print("Decoding...")
    results = []
    for i, model in enumerate(model_fpaths):
        model_name = Path(model).stem
        print(f"On model {model_name}")
        output_fpath = Path(output_dir, f"decoded-{input_name}-{model_name}.txt")
        
        if not args.eval_only:
            translate_parser = _get_parser()
            opt = translate_parser.parse_args([
                '-model', str(model),
                '-src', str(Path(args.data_dir, args.nodes_fname)),
                '-graph', str(Path(args.data_dir, args.graph_fname)),
                '-output', str(output_fpath),
                '-beam_size', '3',
                '-share_vocab',
                '-length_penalty', 'wu',
                '-alpha', args.beam_size,
                #'-verbose',
                '-batch_size', '80',
                '-gpu', '0'
            ])
            translate(opt)

            # detokenize and save
            with open(output_fpath, "r") as infile:
                if not args.tokenizer_path_or_name: # this is the default
                    decoded = [re.sub("(@@ )|(@@ ?$)", "", l.strip()) for l in infile]
                if "t5" in args.tokenizer_path_or_name:
                    decoded = [l.replace(" ", "").replace("▁", " ").strip() for l in infile]
                if "bert" in args.tokenizer_path_or_name:
                    decoded = [l.replace(" ##", "").strip() for l in infile]
            with open(output_fpath, "w") as outfile:
                outfile.write("\n".join(decoded))
        else:
            with open(output_fpath, "r") as infile:
                decoded = [l.strip() for l in infile]
        
        # evaluate
        bleu = sacrebleu.corpus_bleu(decoded, references)
        best_bleu = max(best_bleu, bleu.score)
        results.append((model_name, bleu))
    
    with open(Path(output_dir, f"bleu-{input_name}.txt"), "w") as outfile:
        for model_name, bleu in results:
            if bleu.score == best_bleu:
                print(f"Best BLEU: {bleu.score:0.2f} @ {model_name}")
                model_name = f"{model_name}*"
            outfile.write(f"{model_name:12}: {bleu.format()}\n")
