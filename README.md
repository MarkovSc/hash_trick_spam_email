# Spam Email  Implement of the paper "Collaborative Email-Spam Filtering with the Hashing-Trick"

这篇论文在于在文本的bag of word 或者hash的vectorizer的方式中，如何压缩入user_id 这个维度属性
其实这个可以直接对user_id 做hash之后的concat或 bag of word 的concat，对比的方法为individual classifier ，并不是太明确
但是这个hash的思路，直接嵌入user_idL: word 的hash 在实时模型中还是有意义的。


hash：0.9931
bag of word： 0.9914

- use the common text classifier with the tokenizer
- use the hash method to encode the word
- use the user_id hash method to encode the word











