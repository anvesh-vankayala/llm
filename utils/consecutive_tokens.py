def get_consecutive_tokens(li, window_size=4):
    final_tokens = []
    if len(li) == 0:
        return []
    i = 0
    while i <= len(li)-1:
        j = 1
        while j <= window_size:
            # base case
            if i+j >= len(li):
                final_tokens.append(tuple(li[i:]))
                return final_tokens
            final_tokens.append(tuple(li[i:i+j]))
            j+=1
        i+=1
    return final_tokens

print(get_consesscutive_tokens([1,2,3,4,5]))


import re
from utils.encode_parallel_telugu import encode_tokens_parallel
text = "తెలుగు భాష ఒక ద్రావిడ భాష."

# encoded_tokens = encode_tokens_parallel(text, chunk_size=1_000_000, max_workers=2)
encoded_tokens = [token.encode('utf-8') for token in text]
decoded_tokens = [i.decode('utf-8') for i in encoded_tokens]
print(get_consecutive_tokens(decoded_tokens))

# li = encode_tokens_parallel(text, chunk_size=1_000_000, max_workers=10)
# [print(i.decode('utf-8')) for i in li]
