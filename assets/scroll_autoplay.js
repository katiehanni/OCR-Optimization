function bindAuditRateAutoplay() {
  const wrap = document.getElementById("audit-rate-over-time-wrap");
  if (!wrap || wrap.dataset.autoplayBound === "1") return;
  const plot = wrap.querySelector(".js-plotly-plot");
  if (!plot || typeof window.Plotly === "undefined") return;

  wrap.dataset.autoplayBound = "1";
  let played = false;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (!played && entry.isIntersecting) {
          played = true;
          window.Plotly.animate(plot, null, {
            frame: { duration: 180, redraw: true },
            transition: { duration: 120 },
            fromcurrent: false,
            mode: "immediate",
          });
          observer.disconnect();
        }
      });
    },
    { threshold: 0.45 }
  );

  observer.observe(wrap);
}

function bootAuditAutoplay() {
  bindAuditRateAutoplay();
  const mo = new MutationObserver(() => bindAuditRateAutoplay());
  mo.observe(document.body, { childList: true, subtree: true });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", bootAuditAutoplay);
} else {
  bootAuditAutoplay();
}
