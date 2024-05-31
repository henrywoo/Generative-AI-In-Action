from datasets import load_dataset
import hashlib

data = load_dataset("json", data_files="https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/raw/main/raw/question.jsonl", split="train")

# %% create_dataset.ipynb 11
def format_example(example):
    return {
        "prompt": example["prompt"],
        "prompt_id": int(hashlib.sha256(''.join(example["prompt"]).encode("utf-8")).hexdigest(), 16) % (10 ** 8),
        "category": example["category"],
        "reference": example["reference"],
    }

formatted_ds = data.map(format_example, num_proc=6, remove_columns=data.column_names)
print(formatted_ds)
#
#formatted_ds.push_to_hub("HuggingFaceH4/mt_bench_prompts", split="train")

"""
{"question_id": 126, "category": "coding", "prompt": ["Implement a function to find the median of two sorted arrays of different sizes with O(1) space complexity and O(n) time complexity.", "Does there exist an implementation with better time complexity?"], "reference": ["Carefully check if the given solution is linear complexity.\n\n```\ndef find_median(arr1, arr2):\n    n1 = len(arr1)\n    n2 = len(arr2)\n    if (n1 + n2) == 0:\n        return None\n\n    i, j = 0, 0\n    last_1, last_2 = None, None\n\n    for k in range(1, (n1 + n2) // 2 + 2):\n        last_2 = last_1\n        if j == n2:\n            last_1 = arr1[i]\n            i += 1\n        elif i == n1:\n            last_1 = arr2[j]\n            j += 1\n        elif arr1[i] < arr2[j]:\n            last_1 = arr1[i]\n            i += 1\n        else:\n            last_1 = arr2[j]\n            j += 1\n        \n    if (n1 + n2) % 2 == 1:\n        return last_1\n    else:\n        return (last_1 + last_2) / 2\n```", "There's a binary search solution with O(logn) time complexity.\n\nSample answer:\n```\ndef findMedian(nums1, nums2):\n    total = len(nums1) + len(nums2)\n    if total % 2 == 1:\n        return findKth(nums1, nums2, total // 2 + 1)\n    else:\n        return (findKth(nums1, nums2, total // 2) + findKth(nums1, nums2, total // 2 + 1)) / 2.0\ndef findKth(nums1, nums2, k):\n    if len(nums1) > len(nums2):\n        nums1, nums2 = nums2, nums1\n    if not nums1:\n        return nums2[k-1]\n    if k == 1:\n        return min(nums1[0], nums2[0])\n    i = min(k // 2, len(nums1))\n    j = k - i\n    if nums1[i-1] <= nums2[j-1]:\n        return findKth(nums1[i:], nums2, j) \n    else:\n        return findKth(nums1, nums2[j:], i)\n```"]}

{"question_id": 127, "category": "coding", "prompt": ["Write a function to find the majority element in a given integer array using the Boyer-Moore Voting Algorithm.", "How about finding the top-2 most occurring elements?"], "reference": ["Check if they implement the classical algorithm correctly.\n\nSample answer:\n```\ndef majority_element(arr):\n    count = 0\n    candidate = None\n    # Boyer-Moore Voting Algorithm\n    for num in arr:\n        if count == 0:\n            candidate = num\n        count += (1 if num == candidate else -1)\n    # Verify if the candidate is indeed the majority element\n    if arr.count(candidate) > len(arr) // 2:\n        return candidate\n    else:\n        return None\n```", "There is no simple modification based on the Boyer-Moore Voting Algorithm. Expected answer is to use a hash table.\n\n```\ndef topTwo(nums):\n    # Build a frequency map\n    frequency_map = {}\n    for num in nums:\n        if num in frequency_map:\n            frequency_map[num] += 1\n        else:\n            frequency_map[num] = 1\n\n    # Find the top two most occurring elements\n    most_frequent = sorted(frequency_map.items(), key=lambda x: x[1], reverse=True)[:2]\n\n    return [num for num, _ in most_frequent]\n```"]}

{"question_id": 128, "category": "coding", "prompt": ["A binary tree is full if all of its vertices have either zero or two children. Let B_n denote the number of full binary trees with n vertices. Implement a function to find B_n.", "What if the problem changed from a binary tree to a ternary tree?"], "reference": ["Expected answer is dynamic programming shown below. Some chatbot may answer using Catalan number.\nCheck edge case like when n is even -> return 0.\n\n```python\ndef full_binary_trees(n):\n    if n % 2 == 0:\n        return 0\n    if n == 1:\n        return 1\n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n\n    for i in range(3, n + 1, 2):\n        for j in range(1, i - 1, 2):\n            dp[i] += dp[j] * dp[i - j - 1]\n\n    return dp[n]\n```", "DP is still the expected answer. Catalan number is not correct. Check transition equation carefully.\n\n```python\ndef full_ternary_trees(n):\n    if n % 3 != 1:\n        return 0\n    if n == 1:\n        return 1\n\n    dp = [0] * (n + 1)\n    dp[1] = 1\n\n    for i in range(4, n + 1, 3):\n        for j in range(1, i - 1, 3):\n            for k in range(1, i - j - 1, 3):\n                dp[i] += dp[j] * dp[k] * dp[i - j - k - 1]\n\n    return dp[n]\n```"]}

"""