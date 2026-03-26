import torch
text = "हस्पिटलमा पैसा कसरि तिरे होला? के लाग्छ तिमीलाई?” क्षितिजले यति भन्ने बित्तिकै मैले उसलाई अंगाले| उसले मेरो लागि यति गर्यो, धेरै नै गर्यो| म अहिले क्षितिजमा आफुलाई भेट्दै छु|"

chars = sorted(list(set(text)))
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
# print(stoi)
# print(itos)

data = torch.tensor([stoi[c] for c in text], dtype = torch.long)
# print(data)