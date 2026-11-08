# Dependencies and data access

This source archive contains no participant scans, participant identifiers, AAL3 atlas volumes, pretrained neural-network weights, credentials, or manuscript result tables.

- **ADNI:** obtain data through the applicable authorized access process and follow the dataset's current data-use and publication terms. This package grants no data access or redistribution rights.
- **AAL3 / MNI templates:** obtain the exact atlas/template versions separately, record checksums and transforms, and follow their respective licenses. Synthetic region names are not an anatomical atlas.
- **PyTorch / torchvision:** used for tensor computation, ResNet/FPN, and transformers. The optional pretrained ImageNet weights have their own applicable terms.
- **NumPy / scikit-learn:** used for array processing, splitting, and metrics.
- **Hugging Face transformers:** optional causal-LM interface. Llama and MedGemma weights/access terms are separate from this source package. Model loaders in the supplied reporting and bridge scripts use local files only.
- **pytest:** test runner; the language gradient test constructs a tiny random model without network access.

The encompassing repository's README states an MIT license. This new package does not change upstream dependency, model, atlas, or dataset rights. Authors should retain the appropriate repository license and attribution when redistributing the code under their chosen release process.

API references consulted during implementation:

- [PyTorch MultiheadAttention](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.activation.MultiheadAttention.html): boolean key-padding masks exclude missing memory positions.
- [Transformers generation](https://huggingface.co/docs/transformers/main_classes/text_generation): causal generation and generation configuration.

These references explain interfaces, not scientific validation or reproduction of the manuscript.
