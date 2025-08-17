#!/usr/bin/env python3
import sys
import os
from pathlib import Path
from typing import Iterable, Union

def _get_markdown_text(convert_result: Union[str, object]) -> str:
    if isinstance(convert_result, str):
        return convert_result
    for attr in ("text_content", "markdown", "text"):
        if hasattr(convert_result, attr):
            val = getattr(convert_result, attr)
            if isinstance(val, str):
                return val
    if isinstance(convert_result, dict):
        for key in ("text_content", "markdown", "text", "content"):
            if key in convert_result and isinstance(convert_result[key], str):
                return convert_result[key]
    return str(convert_result)

def iter_pdfs_safely(root: Path) -> Iterable[Path]:
    """
    安全遍历 root 下的所有 .pdf（大小写不敏感）。
    遇到 I/O/权限/坏链接等错误时跳过该目录或文件，继续遍历。
    """
    root = root.resolve()
    def _onerror(e):
        print(f"[walk-skip] {e.filename or ''} -> {e}", file=sys.stderr)

    for dirpath, dirnames, filenames in os.walk(root, onerror=_onerror, followlinks=False):
        # 过滤掉可能出问题的子目录，避免反复触发错误
        safe_dirnames = []
        for d in dirnames:
            full = os.path.join(dirpath, d)
            try:
                # 尝试一次轻量访问，若报错则跳过该子目录
                os.lstat(full)
                safe_dirnames.append(d)
            except OSError as e:
                print(f"[dir-skip] {full} -> {e}", file=sys.stderr)
        # 原地修改，告诉 os.walk 不再深入问题目录
        dirnames[:] = safe_dirnames

        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                fp = Path(dirpath) / fn
                try:
                    # 同样先 lstat 一下，坏链接/坏 inode 会在这里抛异常
                    os.lstat(fp)
                    yield fp
                except OSError as e:
                    print(f"[file-skip] {fp} -> {e}", file=sys.stderr)

def convert_zotero_pdfs_to_markdown(
    src_root: Path = Path("~/Zotero/storage").expanduser(),
    dst_root: Path = Path("~/markdown").expanduser(),
) -> None:
    try:
        from markitdown import MarkItDown
    except Exception:
        print("请先安装 markitdown： pip install 'markitdown[pdf, docx, pptx]'", file=sys.stderr)
        raise

    if not src_root.exists():
        print(f"源目录不存在：{src_root}", file=sys.stderr)
        return

    dst_root.mkdir(parents=True, exist_ok=True)
    md_converter = MarkItDown()

    pdf_iter = list(iter_pdfs_safely(src_root))
    total = len(pdf_iter)
    if total == 0:
        print(f"未在 {src_root} 找到 PDF 文件。")
        return

    print(f"发现 {total} 个 PDF，开始转换…")
    success, failed = 0, 0

    for idx, pdf_path in enumerate(pdf_iter, start=1):
        try:
            rel = pdf_path.relative_to(src_root)
        except ValueError:
            # 极少数情况下路径解析异常，退化为扁平保存
            rel = Path(pdf_path.name)

        out_path = dst_root / rel.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            result = md_converter.convert(str(pdf_path))
            md_text = _get_markdown_text(result)
            header = f"# {pdf_path.stem}\n\n"
            out_path.write_text(header + md_text, encoding="utf-8")
            success += 1
            print(f"[{idx}/{total}] OK  -> {out_path}")
        except Exception as e:
            failed += 1
            print(f"[{idx}/{total}] FAIL -> {pdf_path}\n  原因: {e}", file=sys.stderr)

    print(f"完成：成功 {success}，失败 {failed}。Markdown 已输出至：{dst_root}")

if __name__ == "__main__":
    convert_zotero_pdfs_to_markdown()
