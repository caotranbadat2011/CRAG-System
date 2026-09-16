from __future__ import annotations

import hashlib
import importlib.util
import io
import mimetypes
from pathlib import Path
from typing import Any, Callable

from ..exceptions import ParseError
from ..models import ParsedDocument, TextBlock
from .base import DocumentParser
from .pdf_layout import classify_document_structure, extract_page_layout

PasswordProvider = Callable[[Path], str | bytes | None]


class PdfParser(DocumentParser):
    extensions = (".pdf",)

    def __init__(
        self,
        max_pages: int = 2_000,
        *,
        password: str | bytes | None = None,
        password_provider: PasswordProvider | None = None,
        extract_images: bool = True,
    ) -> None:
        self.max_pages = max_pages
        self.password = password
        self.password_provider = password_provider
        self.extract_images = extract_images

    @property
    def signature(self) -> str:
        backend = "pdfplumber" if importlib.util.find_spec("pdfplumber") else "pypdf"
        return f"pdf-v5|backend={backend}|images={int(self.extract_images)}"

    def parse(self, path: Path, *, source_path: Path | None = None) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ParseError("PDF support requires the 'pypdf' package") from exc

        logical_source = (source_path or path).resolve()
        warnings: list[str] = []
        plumber_pdf: Any | None = None
        encrypted = False
        supplied_password: str | bytes | None = None
        try:
            reader = PdfReader(str(path), strict=False)
            encrypted = bool(reader.is_encrypted)
            if encrypted:
                supplied_password = self.password
                if supplied_password is None and self.password_provider is not None:
                    supplied_password = self.password_provider(logical_source)
                candidates = [supplied_password] if supplied_password is not None else []
                if "" not in candidates:
                    candidates.append("")
                decrypted = False
                for candidate in candidates:
                    try:
                        if reader.decrypt(candidate or ""):
                            decrypted = True
                            break
                    except Exception:
                        continue
                if not decrypted:
                    message = "supplied password was rejected" if supplied_password is not None else "a password is required"
                    raise ParseError(f"Encrypted PDF cannot be opened; {message}: {logical_source}")
            if len(reader.pages) > self.max_pages:
                raise ParseError(f"PDF has {len(reader.pages)} pages; limit is {self.max_pages}")

            plumber_pdf = self._open_pdfplumber(path, supplied_password, warnings)
            blocks: list[TextBlock] = []
            page_metadata: list[dict[str, object]] = []
            image_count = 0
            for page_number, page in enumerate(reader.pages, start=1):
                plumber_page = (
                    plumber_pdf.pages[page_number - 1]
                    if plumber_pdf is not None and page_number <= len(plumber_pdf.pages)
                    else None
                )
                page_blocks, layout = extract_page_layout(
                    page, plumber_page, page_number, logical_source
                )
                images = self._extract_page_images(
                    page, plumber_page, path, logical_source, page_number, warnings
                )
                image_count += len(images)
                graphics_count = self._graphics_operation_count(page)
                layout["graphics_operation_count"] = graphics_count
                layout["image_count"] = len(images)

                extracted_text_chars = sum(
                    sum(character.isalnum() for character in block.text)
                    for block in page_blocks
                )
                for image in images:
                    label = str(image.get("name") or f"page-{page_number}-image")
                    page_blocks.append(
                        TextBlock(
                            f"[Image on page {page_number}: {label}]",
                            "image",
                            page_number,
                            metadata={"image": image, "source": image["source"]},
                        )
                    )

                table_count = layout.get("table_count", 0)
                has_detected_table = (
                    isinstance(table_count, (int, float)) and table_count > 0
                )
                if graphics_count >= 12 and not has_detected_table:
                    page_blocks.append(
                        TextBlock(
                            f"[Vector graphic on page {page_number}]",
                            "graphic",
                            page_number,
                            metadata={
                                "possible_chart_or_diagram": True,
                                "graphics_operation_count": graphics_count,
                                "source": {
                                    "path": str(logical_source),
                                    "page": page_number,
                                    "bbox": [
                                        0.0,
                                        0.0,
                                        float(page.mediabox.width),
                                        float(page.mediabox.height),
                                    ],
                                },
                            },
                        )
                    )
                if extracted_text_chars == 0 and images:
                    warnings.append(
                        f"Page {page_number} contains images but no extractable text"
                    )
                elif not page_blocks:
                    warnings.append(
                        f"Page {page_number} contains no extractable text, image, or vector graphic"
                    )
                blocks.extend(page_blocks)
                page_metadata.append(layout)

            classify_document_structure(blocks)
            fallback_pages = [index for index, layout in enumerate(page_metadata, 1)
                              if layout.get("backend") == "pypdf_text_fallback"]
            if fallback_pages:
                warnings.append(
                    "Recovered missing word spacing with pypdf text on pages "
                    + ", ".join(map(str, fallback_pages))
                    + "; text locations on these pages are page-level only"
                )
            for ordinal, block in enumerate(blocks):
                block.ordinal = ordinal
            metadata: dict[str, object] = {
                str(key).lstrip("/"): str(value)
                for key, value in (reader.metadata or {}).items()
                if value is not None
            }
            metadata.update(
                {
                    "page_count": len(reader.pages),
                    "encrypted": encrypted,
                    "source_path": str(logical_source),
                    "layout_backend": ("mixed" if fallback_pages else
                                       "pdfplumber" if plumber_pdf is not None else "pypdf"),
                    "text_fallback_pages": fallback_pages,
                    "pages": page_metadata,
                    "image_count": image_count,
                }
            )
            return ParsedDocument(logical_source, "application/pdf", blocks, metadata, warnings)
        except ParseError:
            raise
        except Exception as exc:
            raise ParseError(f"Unable to parse PDF: {logical_source}: {exc}") from exc
        finally:
            if plumber_pdf is not None:
                plumber_pdf.close()

    @staticmethod
    def _open_pdfplumber(
        path: Path,
        password: str | bytes | None,
        warnings: list[str],
    ) -> Any | None:
        try:
            import pdfplumber  # type: ignore[import-not-found]
        except ImportError:
            warnings.append(
                "pdfplumber is unavailable; using reduced-fidelity pypdf layout extraction"
            )
            return None
        try:
            if isinstance(password, bytes):
                password = password.decode("utf-8")
            return pdfplumber.open(str(path), password=password or None)
        except Exception as exc:
            warnings.append(f"pdfplumber could not open the PDF; using pypdf fallback: {exc}")
            return None

    def _extract_page_images(
        self,
        page: Any,
        plumber_page: Any | None,
        stored_path: Path,
        source_path: Path,
        page_number: int,
        warnings: list[str],
    ) -> list[dict[str, object]]:
        try:
            image_files = list(page.images)
        except Exception as exc:
            warnings.append(f"Unable to enumerate images on page {page_number}: {exc}")
            return []
        if not image_files:
            return []
        output_dir = stored_path.parent / f"{stored_path.stem}.media"
        if self.extract_images:
            output_dir.mkdir(parents=True, exist_ok=True)
        result: list[dict[str, object]] = []
        for image_index, image_file in enumerate(image_files, start=1):
            try:
                data = image_file.data
                name = Path(str(image_file.name or f"image-{image_index}.bin")).name
                suffix = Path(name).suffix.lower()
                width: int | None = None
                height: int | None = None
                image_format: str | None = None
                try:
                    from PIL import Image

                    with Image.open(io.BytesIO(data)) as image:
                        width, height = image.size
                        image_format = image.format
                        if not suffix and image.format:
                            suffix = f".{image.format.casefold()}"
                except Exception:
                    pass
                suffix = suffix or ".bin"
                digest = hashlib.sha256(data).hexdigest()
                destination = output_dir / f"page-{page_number:04d}-{digest[:20]}{suffix}"
                if self.extract_images and not destination.exists():
                    destination.write_bytes(data)
                plumber_image = None
                if plumber_page is not None:
                    try:
                        plumber_image = plumber_page.images[image_index - 1]
                    except (AttributeError, IndexError, TypeError):
                        pass
                bbox = self._image_bbox(image_file, page, plumber_image)
                result.append(
                    {
                        "name": name,
                        "page": page_number,
                        "index": image_index,
                        "sha256": digest,
                        "content_type": mimetypes.guess_type(f"image{suffix}")[0]
                        or "application/octet-stream",
                        "format": image_format,
                        "width": width,
                        "height": height,
                        "size_bytes": len(data),
                        "extracted_path": str(destination.resolve()) if self.extract_images else None,
                        "source": {
                            "path": str(source_path),
                            "page": page_number,
                            "bbox": bbox,
                        },
                    }
                )
            except Exception as exc:
                warnings.append(
                    f"Unable to extract image {image_index} on page {page_number}: {exc}"
                )
        return result

    @staticmethod
    def _image_bbox(
        image_file: Any, page: Any, plumber_image: dict[str, Any] | None = None
    ) -> list[float]:
        if plumber_image is not None:
            try:
                return [
                    float(plumber_image["x0"]),
                    float(plumber_image["top"]),
                    float(plumber_image["x1"]),
                    float(plumber_image["bottom"]),
                ]
            except (KeyError, TypeError, ValueError):
                pass
        # pypdf exposes image bytes but not placement matrices through page.images.
        # Preserve a page-level bbox rather than inventing inaccurate coordinates.
        return [0.0, 0.0, float(page.mediabox.width), float(page.mediabox.height)]

    @staticmethod
    def _graphics_operation_count(page: Any) -> int:
        try:
            contents = page.get_contents()
            operations = contents.operations if contents is not None else []
            drawing_operators = {b"m", b"l", b"re", b"c", b"v", b"y", b"S", b"s"}
            return sum(1 for _, operator in operations if operator in drawing_operators)
        except Exception:
            return 0
