# This implements an end-to-end process for taking a corpus of text and building a language model for it, based on transfer learning wikitext103
# It saves progress along the way
import numpy as np
from pathlib import Path
import sklearn
from fastai.text import *
import os.path
from shutil import rmtree
import html

class TextProcessorPipeline(object):
  def __init__(self, path, text_dir, read_as_csv=False, pretrained_model_path=None):
    self.path = Path(path)
    self.text_path = path/text_dir
    if pretrained_model_path:
      self.pretrained_model_path = pretrained_model_path
    else:
      self.pretrained_model_path = path/'pretrained'

    self.read_as_csv = read_as_csv

    self.trn_texts_tk = None
    self.val_texts_tk = None
    self.int2tok = None
    self.trn_texts_idx = None
    self.val_texts_idx = None
    
    self.model_path = self.path/'model_dir'
    self.model_path.mkdir(exist_ok=True)
    self.load_progress() # Load saved content from disk

  def save_progress(self):
    np.save(self.model_path/'trn_texts_tk.npy', self.trn_texts_tk)
    np.save(self.model_path/'val_texts_tk.npy', self.val_texts_tk)
    pickle.dump(self.int2tok, open(self.model_path/'int2tok.pkl', 'wb'))
    np.save(self.model_path/'trn_texts_idx.npy', self.trn_texts_idx)
    np.save(self.model_path/'val_texts_idx.npy', self.val_texts_idx)

  def load_progress(self):
    if os.path.isfile(self.model_path/'trn_texts_tk.npy'):
      print("Loading tokenized texts from disk")
      self.trn_texts_tk = np.load(self.model_path/'trn_texts_tk.npy')
    if os.path.isfile(self.model_path/'val_texts_tk.npy'):
      self.val_texts_tk = np.load(self.model_path/'val_texts_tk.npy')

    if os.path.isfile(self.model_path/'int2tok.pkl'):
      print("Loading integer to token mappings from disk")
      self.int2tok = pickle.load(open(self.model_path/'int2tok.pkl', 'rb'))
      if self.int2tok:
        self.tok2int = collections.defaultdict(lambda: 0, { s: i for i, s in enumerate(self.int2tok) })

    if os.path.isfile(self.model_path/'trn_texts_idx.npy'):
      print("Loading integerized texts from disk")
      self.trn_texts_idx = np.load(self.model_path/'trn_texts_idx.npy')
    if os.path.isfile(self.model_path/'val_texts_idx.npy'):
      self.val_texts_idx = np.load(self.model_path/'val_texts_idx.npy')

  def clear_progress(self):
    rmtree(self.model_path)
    self.model_path.mkdir(exist_ok=True)
    self.trn_texts_tk = None
    self.val_texts_tk = None
    self.int2tok = None
    self.trn_texts_idx = None
    self.val_texts_idx = None

  def process_text(self):
    BOS = 'xbos' # Beginning of sentence
    FLD = 'xfld' # Beginning of field

    if self.val_texts_tk is None:
      texts = self.get_texts()
      print("Total texts: ", len(texts))
      trn_texts,val_texts = sklearn.model_selection.train_test_split(texts, test_size=0.1)

      self.trn_texts_tk = self.tokenize(trn_texts)
      self.val_texts_tk = self.tokenize(val_texts)
      self.save_progress()

    if self.int2tok is None:
      self.int2tok, self.tok2int = numericalize_tok(list(self.trn_texts_tk) + list(self.val_texts_tk), max_vocab=60000, min_freq=2)
      self.save_progress()

    if self.trn_texts_idx is None:
      self.trn_texts_idx = [self.tokens_to_ints(text, self.tok2int) for text in self.trn_texts_tk]
      self.val_texts_idx = [self.tokens_to_ints(text, self.tok2int) for text in self.val_texts_tk]
      self.save_progress()

  def get_pretrained_model(self, bptt=70, bs=32, opt_fn = None, drop_scale=1.0):
    """Returns a pretrained language model based on wikitext103
    """
    if opt_fn is None: opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

    PRE_LM_PATH = self.pretrained_model_path/'fwd_wt103.h5'
    wgts = torch.load(PRE_LM_PATH, map_location=lambda storage, loc: storage)

    int2tok_pre = pickle.load((self.pretrained_model_path/'itos_wt103.pkl').open('rb'))
    tok2int_pre = collections.defaultdict(lambda: -1, {v: k for k,v in enumerate(int2tok_pre)})

    # Change the embedding matrix to be setup for the vocab of our dataset, by copying the rows from the pretrained embedding matrix into the correct order
    vs = len(self.int2tok) # Vocab size of our dataset
    em_sz = 400 # Embedding size
    nh = 1150 # Hidden units per layer
    nl = 3 # Number of layers 

    enc_wgts = to_np(wgts['0.encoder.weight'])
    # For tokens that don't have embeddings in the pretrained model, we'll initialise with the mean weights of the other embeddings
    row_m = enc_wgts.mean(0)

    new_emb = np.zeros((vs, em_sz), dtype=np.float32)

    for idx, s in enumerate(self.int2tok):
        idx_pre = tok2int_pre[s]
        if idx_pre >= 0:
            new_emb[idx] = enc_wgts[idx_pre]
        else:
            new_emb[idx] = row_m

    wgts['0.encoder.weight'] = T(new_emb)
    wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_emb))
    wgts['1.decoder.weight'] = T(np.copy(new_emb))

    md = self.get_model_data(bs=bs, bptt=bptt)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15])*drop_scale
    learner = md.get_model(opt_fn, em_sz, nh, nl, dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])

    learner.model.load_state_dict(wgts)

    return learner

  def get_model_data(self, bs=32, bptt=70, sample_frac=1.0):
    vs = len(self.int2tok)

    trn_sample = np.random.choice(self.trn_texts_idx, size=int(len(self.trn_texts_idx) * sample_frac))
    val_sample = np.random.choice(self.val_texts_idx, size=int(len(self.val_texts_idx) * sample_frac))

    trn_dl = LanguageModelLoader(np.concatenate(trn_sample), bs, bptt)
    val_dl = LanguageModelLoader(np.concatenate(val_sample), bs, bptt)
    md = LanguageModelData(self.model_path, 1, vs, trn_dl, val_dl, bs=bs, bptt=bptt)
    return md

  def get_texts(self):
    texts = []
    for fname in (self.path).glob('**/*.*'):
      if self.read_as_csv:
        print("Loading CSV")
        df = pd.read_csv(fname)
        for text in tqdm(df.ix[:,0]):
          fixed_text = self.fixup(text)
          texts.append(fixed_text)        
      else:
        fixed_text = self.fixup(fname.open('r').read())
        texts.append(fixed_text)
    
    return np.array(texts)

  def tokenize(self, texts):
    texts_tk = []
    for text in tqdm(self.chunks(texts, 5000), total=1+len(texts)//5000):
      texts_tk += Tokenizer().proc_all_mp(partition_by_cores(text))

    return texts_tk

  @staticmethod
  def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

  @staticmethod
  def fixup(s): 
    re1 = re.compile('<.*?>')
    re2 = re.compile(r'  +')
    s = s.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re2.sub(' ', html.unescape(re1.sub('',s)))

  @staticmethod
  def tokens_to_ints(toks, tok2int):
    return np.array([tok2int[tok] for tok in toks])