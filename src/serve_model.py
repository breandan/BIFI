from fairseq.models.transformer import TransformerModel
import urllib.parse
from flask import Flask, request
from fairseq import tokenizer


app = Flask(__name__)

trans = TransformerModel.from_pretrained(
  'data/round2-BIFI-part2/model-fixer/',
  checkpoint_file='checkpoint.pt',
  data_name_or_path='data/round2-BIFI-part2/orig_bad/fairseq_preprocess__orig_bad.0',
)

@app.route('/api/text', methods=['GET'])
def get_text():
    url = request.args.get('bifi_translate')
    url_tok = request.args.get('bifi_tokenize')
    url_topk = request.args.get('bifi_topk')
    get_k = request.args.get('k')
    if url:
        decoded_url = urllib.parse.unquote(url)
        output = trans.translate(decoded_url, beam=10, verbose=True)
        return output
    elif url_tok:
        decoded_url = urllib.parse.unquote(url_tok)
        output = trans.decode(trans.encode(decoded_url))
        return output
    elif url_topk:
        decoded_url = urllib.parse.unquote(url_topk)
        k = int(get_k)
        # Retrieve the top 10 most likely token sequences
        # output = trans.generate(decoded_url, beam=10, verbose=True)
        concat = trans.sample_topk([decoded_url], beam=k, verbose=True)
        str_concat = '\n'.join(concat)
        return str_concat

# http://localhost:5000/api/text?bifi_translate=def%20foo%28%29%3A%0A%20%20%20%20print%28%22Hello%2C%20World%21%22%29
if __name__ == '__main__':
    app.run()