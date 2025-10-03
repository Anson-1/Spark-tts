COMMON_PUNCS_PATTERN = '\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\=|\,|\.|\?|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\:|\"|\'|\<|\>|\/|\[|\]|\\|\`|\~|——|—|\，|\。|\、|\《|\》|\？|\；|\‘|\’|\：|\“|\”|\【|\】|\、|\{|\}|\||\·|\！|\￥|\…|\（|\）|\-'   
PUNCTS_PATTERN = '\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\=|\,|\.|\?|\!|\@|\#|\$|\%|\^|\&|\*|\(|\)|\_|\+|\:|\"|\'|\<|\>|\/|\[|\]|\\|\`|\~|——|—|\，|\。|\、|\《|\》|\？|\；|\‘|\’|\：|\“|\”|\【|\】|\、|\{|\}|\||\·|\！|\￥|\…|\（|\）|\-'
FRUNIC= 'àâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇ'
PUNC_TOKEN = [',', '.', '?', '!', '!?', '，', '。', '？', '！', '！？', '…', '—','~','～']

PINYIN_TOKEN = ['ā','á','ǎ','à','ē','é','ě','è','ī','í','ǐ','ì','ō','ó','ǒ','ò','ū','ú','ǔ','ù','ǜ','ǘ','ǚ','ǜ']
PINYIN_PATTERN = ''.join(PINYIN_TOKEN)

#UNICODE ENCODES
UNICODE_CN_BASIC = '\u4e00-\u9fff'
UNICODE_CN_EXTEND = '\u3400-\u4DBF'+'\uF900-\uFAFF'+ '\U00020000-\U0002A6DF'+'\U0002A700-\U0002B73F'+'\U0002B740-\U0002B81F'+'\U0002B820-\U0002CEAF'+'\U0002CEB0-\U0002EBEF'+'\U0002F800-\U0002FA1F'+'\U00030000-\U0003134F'+'\U00031350-\U000323AF'
UNICODE_CJK_PUNCS = '\u3000-\u303F'#	CJK Symbols and Punctuation
UNICODE_CJK_RDC_STK = '\u2E80-\u2EFF'+'\u31C0-\u31EF' +'\u2F00-\u2FDF'# 	CJK Radicals and CJK Strokes and Kangxi Strokes
UNICODE_BPMF = '\u3100-\u312F' + '\u31A0-\u31BF'
# UNICODE_CN_EXTEND = '\U00020000-\U0002A6DF'+'\U0002A700-\U0002B739'+'\U0002B740-\U0002B81D'+'\U0002B820-\U0002CEA1'+'\U0002CEB0-\U0002EBE0'+'\U00030000-\U0003134A'
UNICODE_ASCII_NUMS = '\u0030-\u0039' # equal to 0-9
UNICODE_LATIN_BASIC = '\u0041-\u005a'  + '\u0061-\u007a' # equal to A-Z + a-z
UNICODE_ASCII_PUNCS = '\u0021-\u002F'+'\u003A-\u0040'+'\u005B-\u0060'+'\u007B-\u007E'
UNICODE_EXTEND_PUNCS = '\u00a1-\u00bf'+'\u00d7'+'\u00f7'
UNICODE_OTHER_PUNCS = '\u2E00-\u2E7F'
UNICODE_SPACE = '\u0020'+'\u00a0'+'\u3000' 
UNICODE_LATIN_EXTEND = '\u00c0-\u00d6'+'\u00d8-\u00f6'+'\u00f8-\u00ff'
UNICODE_LATIN_EXTEND_A = '\u0100-\u017f'
UNICODE_LATIN_EXTEND_B = '\u0180-\u024f'
UNICODE_LATIN_EXTEND_ADD = '\u1E00-\u1EFF'
UNICODE_IPA = '\u0250-\u02af'
UNICODE_IPA_SP = '\u02b0-\u02ff'
UNICODE_GRCP_BASIC = '\u0370-\u03FF' #	Greek and Coptic
UNICODE_GR_EXTEND = '\u1f00-\u1fFF'
UNICODE_CY_BASIC ='\u0400-\u04FF' # 	Cyrillic
UNICODE_CY_EXTEND = '\u0500-\u052F'
UNICODE_AR_BASIC = '\u0600-\u06FF' # 	Arabic
UNICODE_AR_EXTEND = '\u0750-\u077F'
UNICODE_TB_BASIC = '\u0F00-\u0FFF' #	Tibetan
UNICODE_GEN_PUNCS = '\u2000-\u206F' # 	General Punctuation
UNICODE_SCRIPTS = '\u2070-\u209F'#	Superscripts and Subscripts
UNICODE_CUR = '\u20A0-\u20CF' #	Currency Symbols
UNICODE_LETSYM = '\u2100-\u214F'#	Letterlike Symbols
UNICODE_MATH_BASIC = '\u2200-\u22FF' #	Mathematical Operators
UNICODE_MATH_EXTEND = '\u27C0-\u27EF'+'\u2980-\u29FF'+'\u2A00-\u2AFF'
UNICODE_FULLWIDTH_LATIN = '\uff21-\uff3a'+'\uff41-\uff5a' # Fullwidth A-Z and a-z
UNICODE_FULLWIDTH_NUMS='\uff10-\uff19'  # Fullwidth 0-9
UNICODE_FULLWIDTH_PUNCS='\uff01-\uffff'+'\uff1a-\uff1f'+'\uff3b-\uff40'+'\uff5b-\uff65'
UNICODE_FULLWIDTH_OTHERS='\uff66-\uffef'
UNICODE_ALL_LATIN=UNICODE_LATIN_BASIC+UNICODE_LATIN_EXTEND+UNICODE_LATIN_EXTEND_A+UNICODE_LATIN_EXTEND_B+UNICODE_LATIN_EXTEND_ADD
UNICODE_ALL_CN = UNICODE_CN_BASIC+UNICODE_CN_EXTEND+UNICODE_CJK_RDC_STK
UNICODE_ALL_PUNCS=UNICODE_CJK_PUNCS+UNICODE_ASCII_PUNCS+UNICODE_EXTEND_PUNCS+UNICODE_OTHER_PUNCS+UNICODE_GEN_PUNCS+UNICODE_FULLWIDTH_PUNCS
UNICODE_ALL_SYM=UNICODE_SCRIPTS+UNICODE_CUR+UNICODE_LETSYM+UNICODE_MATH_BASIC+UNICODE_MATH_EXTEND
UNICODE_EN = f'a-zA-Z{FRUNIC}'