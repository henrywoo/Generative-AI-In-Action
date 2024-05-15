# Span-masked language modeling
from typing import Dict, List
import numpy as np
from transformers import BatchEncoding
from dataclasses import dataclass, field
from transformers import AutoTokenizer
import torch

@dataclass
class DataCollatorForT5MLM:
    """
    [Copied from https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_t5_mlm_flax.py]
    Data collator used for T5 span-masked language modeling.
    It is made sure that after masking the inputs are of length `data_args.max_seq_length` and targets are also of fixed length.
    For more information on how T5 span-masked language modeling works, one can take a look
    at the `official paper <https://arxiv.org/pdf/1910.10683.pdf>`__
    or the `official code for preprocessing <https://github.com/google-research/text-to-text-transfer-transformer/blob/master/t5/data/preprocessors.py>`__ .
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        noise_density (:obj:`float`):
            The probability with which to (randomly) mask tokens in the input.
        mean_noise_span_length (:obj:`float`):
            The average span length of the masked tokens.
        input_length (:obj:`int`):
            The expected input length after masking.
        target_length (:obj:`int`):
            The expected target length after masking.
        pad_token_id: (:obj:`int`):
            The pad token id of the model
        decoder_start_token_id: (:obj:`int):
            The decoder start token id of the model
    """

    tokenizer: AutoTokenizer
    noise_density: float
    mean_noise_span_length: float
    input_length: int
    target_length: int
    pad_token_id: int
    decoder_start_token_id: int = field(default=None)

    def __call__(self, examples: List[Dict[str, np.ndarray]]) -> BatchEncoding:
        # convert list to dict and tensorize input
        batch = BatchEncoding(
            {
                k: np.array([examples[i][k] for i in range(len(examples))])
                for k, v in examples[0].items()
            }
        )

        input_ids = batch["input_ids"]
        batch_size, expandend_input_length = input_ids.shape

        mask_indices = np.asarray(
            [
                self.random_spans_noise_mask(expandend_input_length)
                for i in range(batch_size)
            ]
        )
        labels_mask = ~mask_indices

        input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
        labels_sentinel = self.create_sentinel_ids(labels_mask.astype(np.int8))

        batch["input_ids"] = self.filter_input_ids(input_ids, input_ids_sentinel)
        batch["labels"] = self.filter_input_ids(input_ids, labels_sentinel)

        if batch["input_ids"].shape[-1] != self.input_length:
            raise ValueError(
                f"`input_ids` are incorrectly preprocessed. `input_ids` length is {batch['input_ids'].shape[-1]}, but"
                f" should be {self.input_length}."
            )

        if batch["labels"].shape[-1] != self.target_length:
            raise ValueError(
                f"`labels` are incorrectly preprocessed. `labels` length is {batch['labels'].shape[-1]}, but should be"
                f" {self.target_length}."
            )

        batch = {k: torch.from_numpy(v) for k, v in batch.items()}
        return batch

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[:, 0] = mask_indices[:, 0]

        sentinel_ids = np.where(
            start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices
        )
        sentinel_ids = np.where(
            sentinel_ids != 0, (len(self.tokenizer) - sentinel_ids), 0
        )
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        batch_size = input_ids.shape[0]

        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        # input_ids tokens and sentinel tokens are >= 0, tokens < 0 are
        # masked tokens coming after sentinel tokens and should be removed
        input_ids = input_ids_full[input_ids_full >= 0].reshape((batch_size, -1))
        input_ids = np.concatenate(
            [
                input_ids,
                np.full((batch_size, 1), self.tokenizer.eos_token_id, dtype=np.int32),
            ],
            axis=-1,
        )
        return input_ids

    def random_spans_noise_mask(self, length):
        """This function is copy of `random_spans_helper <https://github.com/google-research/text-to-text-transfer-transformer/blob/84f8bcc14b5f2c03de51bd3587609ba8f6bbd1cd/t5/data/preprocessors.py#L2682>`__ .

        Noise mask consisting of random spans of noise tokens.
        The number of noise tokens and the number of noise spans and non-noise spans
        are determined deterministically as follows:
        num_noise_tokens = round(length * noise_density)
        num_nonnoise_spans = num_noise_spans = round(num_noise_tokens / mean_noise_span_length)
        Spans alternate between non-noise and noise, beginning with non-noise.
        Subject to the above restrictions, all masks are equally likely.

        Args:
            length: an int32 scalar (length of the incoming token sequence)
            noise_density: a float - approximate density of output mask
            mean_noise_span_length: a number

        Returns:
            a boolean tensor with shape [length]
        """

        orig_length = length

        num_noise_tokens = int(np.round(length * self.noise_density))
        # avoid degeneracy by ensuring positive numbers of noise and nonnoise tokens.
        num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
        num_noise_spans = int(np.round(num_noise_tokens / self.mean_noise_span_length))

        # avoid degeneracy by ensuring positive number of noise spans
        num_noise_spans = max(num_noise_spans, 1)
        num_nonnoise_tokens = length - num_noise_tokens

        # pick the lengths of the noise spans and the non-noise spans
        def _random_segmentation(num_items, num_segments):
            """Partition a sequence of items randomly into non-empty segments.
            Args:
                num_items: an integer scalar > 0
                num_segments: an integer scalar in [1, num_items]
            Returns:
                a Tensor with shape [num_segments] containing positive integers that add
                up to num_items
            """
            mask_indices = np.arange(num_items - 1) < (num_segments - 1)
            np.random.shuffle(mask_indices)
            first_in_segment = np.pad(mask_indices, [[1, 0]])
            segment_id = np.cumsum(first_in_segment)
            # count length of sub segments assuming that list is sorted
            _, segment_length = np.unique(segment_id, return_counts=True)
            return segment_length

        noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans)
        nonnoise_span_lengths = _random_segmentation(
            num_nonnoise_tokens, num_noise_spans
        )

        interleaved_span_lengths = np.reshape(
            np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1),
            [num_noise_spans * 2],
        )
        span_starts = np.cumsum(interleaved_span_lengths)[:-1]
        span_start_indicator = np.zeros((length,), dtype=np.int8)
        span_start_indicator[span_starts] = True
        span_num = np.cumsum(span_start_indicator)
        is_noise = np.equal(span_num % 2, 1)

        return is_noise[:orig_length]


def demo_t5_mask():
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer
    # Initialize the data collator
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")  # t5-small
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3,
        input_length=50,
        target_length=50,
        pad_token_id=tokenizer.pad_token_id
    )

    # Generate a noise mask
    length = 100  # Length of the sequence for which to generate the mask
    mask = data_collator.random_spans_noise_mask(length)
    print(mask)
    # Plot the noise mask using matplotlib
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 4))
    plt.bar(range(length), mask, color='blue', width=1.0)
    plt.title('Visualization of Noise Mask')
    plt.xlabel('Token Position')
    plt.ylabel('Masked')
    plt.yticks([0, 1], ['No', 'Yes'])
    plt.show()


def demo_t5_input_target():
    # Initialize the tokenizer and data collator
    tokenizer = AutoTokenizer.from_pretrained("google/t5-v1_1-base")

    # Example input
    sentence = """
    Henry has more than 20 years of working experience in major Silicon Valley technology companies and unicorn
    companies, as well as Wall Street investment banks and hedge funds. Henry is proficient in algorithm, data
    structures, ML/AI, and has ten years of experience in ML/AI development and research. He has worked in 
    Singapore and the United States for ten years each and currently leads the team to develop Generative AI,
    Computer Vision, NLP, LLM related AI products. He interviewed more than 500 candidates, and hosted more than
    300 interview debrief meetings.
    """
    tokenized = tokenizer(sentence, return_tensors="np")

    # Adjust here: reshape the input_ids to remove the batch dimension if it's unnecessary
    input_ids = tokenized['input_ids'].squeeze()
    #from nanoT5.utils.copied_utils import compute_input_and_target_lengths
    #before_mask_input_length, target_length = compute_input_and_target_lengths(116, 0.15, 3)
    data_collator = DataCollatorForT5MLM(
        tokenizer=tokenizer,
        noise_density=0.15,
        mean_noise_span_length=3,
        input_length=106,
        target_length=24,
        pad_token_id=tokenizer.pad_token_id,
        decoder_start_token_id=tokenizer.pad_token_id  # assuming pad token id for simplicity
    )
    examples = [{'input_ids': input_ids}]
    # Process the example through the data collator
    batch = data_collator(examples)

    # Decode the tokens to show text for input_ids and labels
    input_text = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=False)
    label_text = tokenizer.decode(batch['labels'][0], skip_special_tokens=False)

    print("Input Text (Masked):", input_text)
    print("Target Text (Labels):", label_text)


if __name__ == "__main__":
    import random
    # Set the seeds for reproducibility
    seed_val = 42  # Choose any integer
    random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)  # If using GPU
    demo_t5_input_target()