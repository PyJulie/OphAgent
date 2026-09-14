// Language Switcher UI Component
// Injects a language selection button into the sidebar

const i18nUI = (() => {
  const SELECTOR_ID = 'language-selector';
  const BUTTON_CLASS = 'language-toggle-btn';
  const DROPDOWN_CLASS = 'language-dropdown';

  function createLanguageSelector() {
    const container = document.createElement('div');
    container.id = SELECTOR_ID;
    container.className = 'language-selector';
    container.setAttribute('title', i18n.get('language.selectLanguage'));

    const button = document.createElement('button');
    button.type = 'button';
    button.className = BUTTON_CLASS;
    button.setAttribute('aria-label', i18n.get('language.selectLanguage'));
    button.setAttribute('aria-haspopup', 'true');
    button.setAttribute('aria-expanded', 'false');
    button.innerHTML = `
      <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z" fill="currentColor"/>
      </svg>
      <span class="language-label">${i18n.getLanguageName()}</span>
    `;

    const dropdown = document.createElement('div');
    dropdown.className = DROPDOWN_CLASS;
    dropdown.setAttribute('role', 'menu');
    dropdown.hidden = true;

    const languages = i18n.getSupportedLanguages();
    const currentLang = i18n.getCurrentLanguage();

    for (const lang of languages) {
      const option = document.createElement('button');
      option.type = 'button';
      option.className = 'language-option';
      option.setAttribute('role', 'menuitem');
      option.setAttribute('data-lang', lang.code);
      if (lang.code === currentLang) {
        option.setAttribute('aria-current', 'true');
        option.classList.add('active');
      }
      option.innerHTML = `
        <span class="language-name">${lang.nativeName}</span>
        ${lang.code === currentLang ? '<span class="language-check" aria-hidden="true">✓</span>' : ''}
      `;
      option.onclick = async () => {
        await switchLanguage(lang.code);
      };
      dropdown.appendChild(option);
    }

    button.onclick = (e) => {
      e.stopPropagation();
      const isOpen = !dropdown.hidden;
      dropdown.hidden = isOpen;
      button.setAttribute('aria-expanded', String(!isOpen));
    };

    container.appendChild(button);
    container.appendChild(dropdown);

    // Close dropdown on outside click
    document.addEventListener('click', (e) => {
      if (!container.contains(e.target) && !dropdown.hidden) {
        dropdown.hidden = true;
        button.setAttribute('aria-expanded', 'false');
      }
    });

    return container;
  }

  async function switchLanguage(langCode) {
    const success = await i18n.setLanguage(langCode);
    if (success) {
      // Update UI immediately
      updateUILanguage();
      
      // Dispatch custom event for other components to listen to
      window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: langCode } }));
    }
  }

  function updateUILanguage() {
    const selector = document.getElementById(SELECTOR_ID);
    if (!selector) return;

    // Update button text
    const button = selector.querySelector(`.${BUTTON_CLASS}`);
    if (button) {
      const label = button.querySelector('.language-label');
      if (label) {
        label.textContent = i18n.getLanguageName();
      }
    }

    // Update dropdown options
    const options = selector.querySelectorAll('.language-option');
    const currentLang = i18n.getCurrentLanguage();
    for (const option of options) {
      const lang = option.getAttribute('data-lang');
      if (lang === currentLang) {
        option.setAttribute('aria-current', 'true');
        option.classList.add('active');
        let check = option.querySelector('.language-check');
        if (!check) {
          check = document.createElement('span');
          check.className = 'language-check';
          check.setAttribute('aria-hidden', 'true');
          check.textContent = '✓';
          option.appendChild(check);
        }
      } else {
        option.removeAttribute('aria-current');
        option.classList.remove('active');
        const check = option.querySelector('.language-check');
        if (check) check.remove();
      }
    }
  }

  function injectStylesheet() {
    const style = document.createElement('style');
    style.textContent = `
      /* Language Selector Styles */
      #language-selector {
        position: relative;
        margin: 8px 0;
        padding: 0 8px;
      }

      .language-toggle-btn {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 10px 12px;
        background: var(--bg-tertiary, #f0f0f0);
        border: 1px solid var(--border-primary, #e0e0e0);
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        color: var(--text-primary, #333);
        transition: all 0.2s ease;
      }

      .language-toggle-btn:hover {
        background: var(--bg-secondary, #e8e8e8);
        border-color: var(--border-secondary, #d0d0d0);
      }

      .language-toggle-btn:active {
        background: var(--bg-primary, #f8f8f8);
      }

      .language-toggle-btn svg {
        flex-shrink: 0;
      }

      .language-label {
        flex: 1;
        text-align: left;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
      }

      .language-dropdown {
        position: absolute;
        top: 100%;
        left: 8px;
        right: 8px;
        margin-top: 4px;
        background: var(--bg-primary, white);
        border: 1px solid var(--border-primary, #e0e0e0);
        border-radius: 6px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        overflow: hidden;
      }

      .language-option {
        display: flex;
        align-items: center;
        gap: 8px;
        width: 100%;
        padding: 10px 12px;
        background: none;
        border: none;
        border-radius: 0;
        cursor: pointer;
        font-size: 14px;
        color: var(--text-primary, #333);
        text-align: left;
        transition: background-color 0.15s ease;
      }

      .language-option:hover {
        background-color: var(--bg-tertiary, #f5f5f5);
      }

      .language-option.active {
        background-color: var(--bg-tertiary, #f5f5f5);
        font-weight: 600;
      }

      .language-option:not(:last-child) {
        border-bottom: 1px solid var(--border-primary, #e0e0e0);
      }

      .language-name {
        flex: 1;
      }

      .language-check {
        flex-shrink: 0;
        color: var(--success, #4CAF50);
        font-weight: bold;
      }

      /* Dark mode support */
      @media (prefers-color-scheme: dark) {
        .language-toggle-btn {
          background: var(--bg-tertiary, #2a2a2a);
          border-color: var(--border-primary, #444);
          color: var(--text-primary, #e0e0e0);
        }

        .language-toggle-btn:hover {
          background: var(--bg-secondary, #333);
          border-color: var(--border-secondary, #555);
        }

        .language-dropdown {
          background: var(--bg-primary, #1e1e1e);
          border-color: var(--border-primary, #444);
        }

        .language-option {
          color: var(--text-primary, #e0e0e0);
        }

        .language-option:hover {
          background-color: var(--bg-tertiary, #2a2a2a);
        }

        .language-option.active {
          background-color: var(--bg-tertiary, #2a2a2a);
        }
      }
    `;
    document.head.appendChild(style);
  }

  async function init() {
    // Inject styles
    injectStylesheet();

    // Wait for DOM to be ready
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', insertSelector);
    } else {
      insertSelector();
    }
  }

  function insertSelector() {
    const sidebar = document.querySelector('.sidebar-foot');
    if (!sidebar) {
      console.warn('Could not find sidebar-foot element');
      return;
    }

    // Create selector
    const selector = createLanguageSelector();
    
    // Insert before the existing hint element
    const hint = sidebar.querySelector('.hint');
    if (hint) {
      hint.parentNode.insertBefore(selector, hint);
    } else {
      sidebar.insertBefore(selector, sidebar.firstChild);
    }
  }

  return {
    init,
    switchLanguage,
  };
})();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => i18nUI.init());
} else {
  i18nUI.init();
}
