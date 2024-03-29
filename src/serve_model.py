from fairseq.models.transformer import TransformerModel
import urllib.parse
from flask import Flask, request
from fairseq import tokenizer


app = Flask(__name__)

breaker = TransformerModel.from_pretrained(
  'data/round2-BIFI-part1/model-breaker/',
  checkpoint_file='checkpoint.pt',
  data_name_or_path='data/round2-BIFI-part1/orig_good/fairseq_preprocess__orig_good.9',
)

fixer = TransformerModel.from_pretrained(
  'data/round2-BIFI-part2/model-fixer/',
  checkpoint_file='checkpoint.pt',
  data_name_or_path='data/round2-BIFI-part2/orig_bad/fairseq_preprocess__orig_bad.4',
)

@app.route('/api/text', methods=['GET'])
def get_text():
    url = request.args.get('bifi_translate')
    url_tok = request.args.get('bifi_tokenize')
    url_topk = request.args.get('bifi_topk')
    get_k = request.args.get('k')
    url_break = request.args.get('bifi_break')
    if url:
        decoded_url = urllib.parse.unquote(url)
        output = fixer.translate(decoded_url, beam=10, verbose=True)
        return output
    elif url_tok:
        decoded_url = urllib.parse.unquote(url_tok)
        output = fixer.decode(fixer.encode(decoded_url))
        return output
    elif url_topk:
        decoded_url = urllib.parse.unquote(url_topk)
        k = int(get_k)
        # Retrieve the top 10 most likely token sequences
        # output = trans.generate(decoded_url, beam=10, verbose=True)
        concat = fixer.sample_topk([decoded_url], beam=k, verbose=True)
        str_concat = '\n'.join(concat)
        return str_concat
    elif url_break:
        decoded_url = urllib.parse.unquote(url_break)
        output = breaker.translate(decoded_url, beam=10, verbose=True)
        return output

# http://localhost:5000/api/text?bifi_break=test
if __name__ == '__main__':
    app.run()