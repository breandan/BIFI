from fairseq.models.transformer import TransformerModel
import urllib.parse
from flask import Flask, request


app = Flask(__name__)

trans = TransformerModel.from_pretrained(
  'data/round2-BIFI-part2/model-fixer/',
  checkpoint_file='checkpoint.pt',
  data_name_or_path='data/round2-BIFI-part2/orig_bad/fairseq_preprocess__orig_bad.0',
)

@app.route('/api/text', methods=['GET'])
def get_text():
    url = request.args.get('bifi')
    decoded_url = urllib.parse.unquote(url)
    output = trans.translate(decoded_url)
    return output


if __name__ == '__main__':
    app.run()