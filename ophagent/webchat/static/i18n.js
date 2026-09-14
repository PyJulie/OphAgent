// i18n (Internationalization) system for OphAgent Web UI
// Supports multiple languages with easy switching

const i18n = (() => {
  let currentLanguage = localStorage.getItem('ophagent.language') || 'en';
  let translations = {};
  let loadedLanguages = new Set();

  const SUPPORTED_LANGUAGES = {
    en: { name: 'English', nativeName: 'English' },
    uk: { name: 'Ukrainian', nativeName: 'Українська' },
  };

  async function loadLanguage(lang) {
    if (loadedLanguages.has(lang)) {
      return translations[lang];
    }

    try {
      const response = await fetch(`/static/i18n/${lang}.json`);
      if (!response.ok) {
        console.error(`Failed to load language file: ${lang}`);
        return null;
      }
      const data = await response.json();
      translations[lang] = data;
      loadedLanguages.add(lang);
      return data;
    } catch (error) {
      console.error(`Error loading language: ${lang}`, error);
      return null;
    }
  }

  function get(key, defaultValue = key) {
    const keys = key.split('.');
    let value = translations[currentLanguage];

    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        return defaultValue;
      }
    }

    return value || defaultValue;
  }

  function getCurrentLanguage() {
    return currentLanguage;
  }

  function getLanguageName(lang = currentLanguage) {
    return SUPPORTED_LANGUAGES[lang]?.nativeName || lang;
  }

  async function setLanguage(lang) {
    if (!SUPPORTED_LANGUAGES[lang]) {
      console.warn(`Unsupported language: ${lang}`);
      return false;
    }

    const loaded = await loadLanguage(lang);
    if (loaded) {
      currentLanguage = lang;
      localStorage.setItem('ophagent.language', lang);
      return true;
    }
    return false;
  }

  function getSupportedLanguages() {
    return Object.entries(SUPPORTED_LANGUAGES).map(([code, info]) => ({
      code,
      name: info.name,
      nativeName: info.nativeName,
    }));
  }

  // Initialize: load current language
  async function initialize() {
    const loaded = await loadLanguage(currentLanguage);
    if (!loaded && currentLanguage !== 'en') {
      // Fallback to English if current language fails to load
      await loadLanguage('en');
      currentLanguage = 'en';
      localStorage.setItem('ophagent.language', 'en');
    }
    return loaded !== null;
  }

  return {
    get,
    getCurrentLanguage,
    getLanguageName,
    setLanguage,
    getSupportedLanguages,
    initialize,
    SUPPORTED_LANGUAGES,
  };
})();

// Expose globally for use in HTML/JS
window.i18n = i18n;
