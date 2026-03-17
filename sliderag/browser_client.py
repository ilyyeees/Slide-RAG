#!/usr/bin/env python3
"""Playwright-based browser automation for Google Gemini."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from playwright.sync_api import (
    BrowserContext,
    ElementHandle,
    Page,
    Playwright,
    TimeoutError as PlaywrightTimeout,
    sync_playwright,
)

logger = logging.getLogger("sliderag.browser_client")


class GeminiBrowserClient:
    """Browser automation client for the Gemini web app."""

    GEMINI_URL = "https://gemini.google.com/app"

    INPUT_SELECTORS = [
        'div.ql-editor[contenteditable="true"]',
        'div[contenteditable="true"][aria-label*="Enter a prompt"]',
        'div[contenteditable="true"][aria-label*="prompt"]',
        'div[contenteditable="true"][role="textbox"]',
        'rich-textarea div[contenteditable="true"]',
        'textarea[aria-label*="prompt"]',
        'textarea[aria-label*="Enter"]',
        '.text-input-field [contenteditable="true"]',
        '.input-area [contenteditable="true"]',
        'div[contenteditable="true"]',
    ]

    SEND_BUTTON_SELECTORS = [
        'button[aria-label="Send message"]',
        'button[aria-label*="Send"]',
        'button[aria-label*="send"]',
        'button[data-tooltip*="Send"]',
        'button[mattooltip*="Send"]',
        'button.send-button',
        '.input-area button[aria-label*="end"]',
        "div.input-buttons button:last-child",
    ]

    RESPONSE_SELECTORS = [
        "message-content.model-response-text",
        "model-response message-content",
        ".model-response-text .markdown-main-panel",
        "message-content .markdown-main-panel",
        ".response-container message-content",
        'div[data-message-author-role="model"] .message-content',
        ".conversation-container model-response",
        "model-response",
        "message-content",
        ".markdown-main-panel",
    ]

    NEW_CHAT_SELECTORS = [
        'button[aria-label="New chat"]',
        'button[aria-label*="New chat"]',
        'a[aria-label*="New chat"]',
        'button[data-tooltip*="New chat"]',
        ".new-chat-button",
        'a[href="/app"]',
    ]

    def __init__(
        self,
        profile_dir: str = "./gemini_browser_profile",
        headless: bool = False,
        response_timeout: int = 600,
        inter_request_delay: int = 15,
        gemini_url: Optional[str] = None,
    ):
        self.profile_dir = Path(profile_dir).resolve()
        self.headless = headless
        self.response_timeout = response_timeout
        self.inter_request_delay = inter_request_delay
        self.gemini_url = gemini_url or self.GEMINI_URL

        self._playwright: Optional[Playwright] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

        self._working_input_selector: Optional[str] = None
        self._working_send_selector: Optional[str] = None
        self._working_response_selector: Optional[str] = None
        self._custom_selectors = self._load_custom_selectors()
        self._request_count = 0

    def _load_custom_selectors(self) -> dict:
        selectors_file = self.profile_dir / "selectors.json"
        if selectors_file.exists():
            try:
                with open(selectors_file, "r", encoding="utf-8") as handle:
                    custom = json.load(handle)
                logger.info("Loaded custom selectors from %s", selectors_file)
                return custom
            except Exception as exc:
                logger.warning("Failed to load custom selectors: %s", exc)
        return {}

    def _save_working_selectors(self) -> None:
        selectors_file = self.profile_dir / "selectors.json"
        working = {}
        if self._working_input_selector:
            working["input_area"] = self._working_input_selector
        if self._working_send_selector:
            working["send_button"] = self._working_send_selector
        if self._working_response_selector:
            working["response_container"] = self._working_response_selector

        if working:
            try:
                self.profile_dir.mkdir(parents=True, exist_ok=True)
                with open(selectors_file, "w", encoding="utf-8") as handle:
                    json.dump(working, handle, indent=2)
            except Exception as exc:
                logger.debug("Could not save selectors: %s", exc)

    def _get_selectors(self, category: str) -> List[str]:
        selectors: List[str] = []
        cache_map = {
            "input_area": self._working_input_selector,
            "send_button": self._working_send_selector,
            "response_container": self._working_response_selector,
        }
        cached = cache_map.get(category)
        if cached:
            selectors.append(cached)

        custom = self._custom_selectors.get(category)
        if isinstance(custom, str) and custom not in selectors:
            selectors.append(custom)
        elif isinstance(custom, list):
            for selector in custom:
                if selector not in selectors:
                    selectors.append(selector)

        default_map = {
            "input_area": self.INPUT_SELECTORS,
            "send_button": self.SEND_BUTTON_SELECTORS,
            "response_container": self.RESPONSE_SELECTORS,
            "new_chat": self.NEW_CHAT_SELECTORS,
        }
        for selector in default_map.get(category, []):
            if selector not in selectors:
                selectors.append(selector)
        return selectors

    def initialize(self) -> None:
        """Launch the persistent browser context and navigate to Gemini."""
        logger.info("Launching browser with persistent profile...")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            user_data_dir=str(self.profile_dir),
            headless=self.headless,
            viewport={"width": 1440, "height": 900},
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
                "--disable-popup-blocking",
            ],
            ignore_default_args=["--enable-automation"],
            user_agent=(
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        )

        self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
        self._page.goto(self.gemini_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)
        self._ensure_logged_in()
        self._dismiss_dialogs()
        logger.info("Browser client initialized and ready")

    def _ensure_logged_in(self) -> None:
        max_login_wait = 300
        current_url = self._page.url
        logger.info("Current URL: %s", current_url)

        if "gemini.google.com" in current_url and "/app" in current_url:
            try:
                self._find_element(
                    self._get_selectors("input_area"),
                    timeout=10000,
                    description="input area",
                )
                logger.info("Already logged into Gemini")
                return
            except Exception:
                logger.debug("Gemini URL loaded but input area not ready yet")

        print("\n" + "=" * 70)
        print("  MANUAL LOGIN REQUIRED")
        print("=" * 70)
        print()
        print("  1. Log into your Google account in the browser")
        print("  2. Open gemini.google.com if needed")
        print("  3. Make sure the Gemini chat UI is visible")
        print("  4. Return here and press ENTER")
        print()
        print(f"  Waiting up to {max_login_wait}s for login...")
        print("=" * 70)
        input("\n  Press ENTER when Gemini is ready... ")

        self._page.goto(self.gemini_url, wait_until="domcontentloaded", timeout=60000)
        time.sleep(4)
        try:
            self._find_element(
                self._get_selectors("input_area"),
                timeout=15000,
                description="input area (post-login verification)",
            )
            logger.info("Login verified")
        except Exception:
            logger.warning(
                "Could not verify login automatically. Proceeding anyway; use --diagnose if needed."
            )

    def _dismiss_dialogs(self) -> None:
        selectors = [
            'button[aria-label="Close"]',
            'button[aria-label="Dismiss"]',
            'button[aria-label="Got it"]',
            'button:has-text("Got it")',
            'button:has-text("I agree")',
            'button:has-text("Accept all")',
            'button:has-text("Accept")',
            'button:has-text("Continue")',
            'button:has-text("OK")',
            'button:has-text("No thanks")',
        ]
        for selector in selectors:
            try:
                element = self._page.query_selector(selector)
                if element and element.is_visible():
                    element.click()
                    time.sleep(1)
            except Exception:
                continue

    def _find_element(
        self,
        selectors: List[str],
        timeout: int = 10000,
        description: str = "element",
    ) -> ElementHandle:
        errors = []
        per_selector_timeout = min(timeout, 3000)

        for selector in selectors:
            try:
                element = self._page.wait_for_selector(
                    selector,
                    timeout=per_selector_timeout,
                    state="visible",
                )
                if element:
                    return element
            except PlaywrightTimeout:
                errors.append(f"  {selector} -> timeout")
            except Exception as exc:
                errors.append(f"  {selector} -> {type(exc).__name__}: {exc}")

        for selector in selectors:
            try:
                element = self._page.query_selector(selector)
                if element and element.is_visible():
                    return element
            except Exception:
                continue

        for selector in selectors:
            try:
                handle = self._page.evaluate_handle(
                    """(sel) => {
                        let el = document.querySelector(sel);
                        if (el) return el;
                        const allHosts = document.querySelectorAll('*');
                        for (const host of allHosts) {
                            if (host.shadowRoot) {
                                el = host.shadowRoot.querySelector(sel);
                                if (el) return el;
                            }
                        }
                        return null;
                    }""",
                    selector,
                )
                element = handle.as_element()
                if element:
                    return element
            except Exception:
                continue

        error_message = (
            f"Could not find {description}. Tried {len(selectors)} selectors:\n"
            + "\n".join(errors)
        )
        logger.error(error_message)
        raise RuntimeError(error_message)

    def _insert_text(self, text: str) -> None:
        selectors = self._get_selectors("input_area")
        input_el = self._find_element(selectors, description="input area")

        for selector in selectors:
            try:
                if self._page.query_selector(selector):
                    self._working_input_selector = selector
                    break
            except Exception:
                continue

        input_el.click()
        time.sleep(0.3)
        self._page.keyboard.press("Control+a")
        self._page.keyboard.press("Backspace")
        time.sleep(0.2)

        try:
            self._page.keyboard.insert_text(text)
            time.sleep(0.5)
            try:
                current = input_el.inner_text()
                if current and len(current.strip()) > 20:
                    return
            except Exception:
                return
        except Exception as exc:
            logger.debug("keyboard.insert_text failed: %s", exc)

        try:
            input_el.click()
            time.sleep(0.2)
            self._page.evaluate(
                """(text) => {
                    document.execCommand('selectAll', false, null);
                    document.execCommand('insertText', false, text);
                }""",
                text,
            )
            time.sleep(0.5)
            return
        except Exception as exc:
            logger.debug("execCommand failed: %s", exc)

        try:
            input_el.evaluate(
                """(el, text) => {
                    el.focus();
                    el.innerText = text;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                }""",
                text,
            )
            time.sleep(0.5)
            return
        except Exception as exc:
            logger.debug("Direct innerText set failed: %s", exc)

        try:
            input_el.click()
            time.sleep(0.2)
            self._page.evaluate(
                """(text) => {
                    const ta = document.createElement('textarea');
                    ta.value = text;
                    ta.style.position = 'fixed';
                    ta.style.left = '-9999px';
                    ta.style.top = '-9999px';
                    document.body.appendChild(ta);
                    ta.select();
                    document.execCommand('copy');
                    document.body.removeChild(ta);
                }""",
                text,
            )
            time.sleep(0.3)
            self._page.keyboard.press("Control+v")
            time.sleep(0.5)
            return
        except Exception as exc:
            logger.debug("Clipboard paste failed: %s", exc)

        logger.warning("Falling back to slow typing for a long Gemini prompt.")
        input_el.click()
        time.sleep(0.2)
        input_el.type(text, delay=1)
        time.sleep(0.5)

    def _submit_prompt(self) -> None:
        for selector in self._get_selectors("send_button"):
            try:
                button = self._page.wait_for_selector(selector, timeout=3000, state="visible")
                if not button:
                    continue
                if button.get_attribute("disabled") or button.get_attribute("aria-disabled") == "true":
                    continue
                button.click()
                self._working_send_selector = selector
                time.sleep(1)
                return
            except Exception:
                continue
        self._page.keyboard.press("Enter")
        time.sleep(1)

    def _select_pro_model(self) -> None:
        logger.debug("Selecting Gemini Pro model...")
        try:
            fast_elements = self._page.query_selector_all(
                'span:has-text("Fast"), div:has-text("Fast")'
            )
            for element in fast_elements:
                if element.is_visible() and element.inner_text().strip() == "Fast":
                    element.click()
                    time.sleep(1)
                    break
            else:
                return

            pro_elements = self._page.query_selector_all(
                'span.gds-title-m:has-text("Pro"), span:has-text("Pro")'
            )
            for element in pro_elements:
                if element.is_visible() and "Pro" in element.inner_text():
                    element.click()
                    time.sleep(1)
                    return
            self._page.mouse.click(0, 0)
            time.sleep(0.5)
        except Exception as exc:
            logger.debug("Could not change model to Gemini Pro: %s", exc)

    def _start_new_chat(self) -> None:
        logger.debug("Starting new chat...")
        for selector in self._get_selectors("new_chat"):
            try:
                button = self._page.query_selector(selector)
                if button and button.is_visible():
                    button.click()
                    time.sleep(3)
                    self._dismiss_dialogs()
                    return
            except Exception:
                continue

        self._page.goto(self.gemini_url, wait_until="domcontentloaded", timeout=30000)
        time.sleep(4)
        self._dismiss_dialogs()

    def _wait_for_response(self) -> str:
        logger.info("Waiting for Gemini response...")
        time.sleep(20)

        last_text = ""
        stable_count = 0
        stable_threshold = 4
        check_interval = 2.0
        start_time = time.time()
        empty_count = 0
        max_empty_checks = 45

        while time.time() - start_time < self.response_timeout:
            current_text = self._extract_latest_response()
            if not current_text or len(current_text.strip()) < 5:
                empty_count += 1
                if empty_count > max_empty_checks:
                    logger.warning("No Gemini response appeared after extended wait")
                    break
                time.sleep(check_interval)
                continue

            empty_count = 0
            if current_text == last_text:
                stable_count += 1
                if stable_count >= stable_threshold:
                    elapsed = time.time() - start_time
                    logger.info(
                        "Gemini response complete: %s chars in %.1fs",
                        len(current_text),
                        elapsed,
                    )
                    return current_text
            else:
                stable_count = 0
                last_text = current_text

            time.sleep(check_interval)

        return last_text

    def _extract_latest_response(self) -> str:
        for selector in self._get_selectors("response_container"):
            try:
                elements = self._page.query_selector_all(selector)
                if elements:
                    last_element = elements[-1]
                    text = last_element.inner_text()
                    if text and len(text.strip()) > 5:
                        self._working_response_selector = selector
                        return text.strip()
            except Exception:
                continue

        fallback_selectors = [
            '[data-turn-role="model"]',
            ".model-turn",
            ".assistant-message",
            '[role="article"]',
        ]
        for selector in fallback_selectors:
            try:
                elements = self._page.query_selector_all(selector)
                if elements:
                    text = elements[-1].inner_text()
                    if text and len(text.strip()) > 5:
                        return text.strip()
            except Exception:
                continue
        return ""

    def send_prompt(self, prompt: str) -> str:
        self._request_count += 1
        logger.info("--- Gemini Request #%s (%s chars) ---", self._request_count, len(prompt))

        if self._request_count > 1:
            logger.info(
                "Rate limiting: waiting %ss between requests...",
                self.inter_request_delay,
            )
            time.sleep(self.inter_request_delay)

        self._start_new_chat()
        self._select_pro_model()
        time.sleep(2)
        self._insert_text(prompt)
        self._submit_prompt()
        response = self._wait_for_response()
        if not response:
            raise RuntimeError(
                "No Gemini response received. The prompt may have been blocked or selectors may need updating."
            )
        self._save_working_selectors()
        return response

    def close(self) -> None:
        logger.info("Closing browser client...")
        try:
            if self._context:
                self._context.close()
        except Exception as exc:
            logger.debug("Error closing context: %s", exc)
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            logger.debug("Error stopping Playwright: %s", exc)
        self._page = None
        self._context = None
        self._playwright = None

    def diagnose(self) -> None:
        print("\n" + "=" * 70)
        print("  GEMINI BROWSER CLIENT DIAGNOSTICS")
        print("=" * 70)
        print(f"\n  Current URL: {self._page.url}")
        print(f"  Page title:  {self._page.title()}")

        categories = {
            "Input Area": self._get_selectors("input_area"),
            "Send Button": self._get_selectors("send_button"),
            "Response Container": self._get_selectors("response_container"),
            "New Chat Button": self._get_selectors("new_chat"),
        }
        for name, selectors in categories.items():
            print(f"\n  --- {name} ---")
            found = False
            for selector in selectors:
                try:
                    element = self._page.query_selector(selector)
                    if element:
                        visible = element.is_visible()
                        preview = (element.inner_text() or "")[:60].replace("\n", " ")
                        status = "VISIBLE" if visible else "HIDDEN"
                        print(f"    {status}: {selector}")
                        if preview:
                            print(f"             Text: '{preview}'")
                        found = True
                except Exception:
                    continue
            if not found:
                print("    No working selector found")

        print(f"\n  --- All contenteditable elements ---")
        try:
            editables = self._page.query_selector_all('[contenteditable="true"]')
            if editables:
                for index, element in enumerate(editables):
                    tag = element.evaluate("el => el.tagName")
                    classes = element.get_attribute("class") or ""
                    aria = element.get_attribute("aria-label") or ""
                    role = element.get_attribute("role") or ""
                    print(
                        f"    [{index}] <{tag}> class='{classes[:50]}' "
                        f"aria-label='{aria[:50]}' role='{role}'"
                    )
            else:
                print("    No contenteditable elements found")
        except Exception as exc:
            print(f"    Error: {exc}")

        print("\n" + "=" * 70)
        print("  To override selectors, edit:")
        print(f"     {self.profile_dir / 'selectors.json'}")
        print("=" * 70 + "\n")
