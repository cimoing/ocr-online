import { createApp } from "vue";
import App from "./App.vue";
// CSS lives in the module graph so dev and build resolve it identically
// (an index.html <link> with %BASE_URL% gets double-prefixed by dev's
// base rewriting and 404s).
import "./style.css";

createApp(App).mount("#app");
