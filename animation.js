// Treat each loop as a pausable teaching figure. Offscreen figures do not play.
(() => {
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  document.querySelectorAll('[data-transport]').forEach((player) => {
    const video = player.querySelector('video');
    const toggle = player.querySelector('[data-toggle]');
    const stages = [...player.querySelectorAll('[data-stage]')];
    let wantsPlayback = !reduceMotion.matches;
    let visible = false;
    video.muted = true;
    video.controls = false;
    player.querySelector('.transport-controls').hidden = false;
    const updateButton = () => {
      toggle.textContent = video.paused ? 'Play animation' : 'Pause';
      toggle.setAttribute('aria-label', `${video.paused ? 'Play' : 'Pause'} CAST animation`);
    };
    const play = () => video.play().catch(updateButton);
    const sync = () => {
      if (visible && wantsPlayback) play();
      else video.pause();
      updateButton();
    };
    toggle.addEventListener('click', () => {
      wantsPlayback = video.paused;
      sync();
    });
    player.querySelector('[data-replay]')?.addEventListener('click', () => {
      video.currentTime = 0;
      wantsPlayback = true;
      sync();
    });
    stages.forEach((button) => button.addEventListener('click', () => {
      wantsPlayback = false;
      video.pause();
      video.currentTime = Number(button.dataset.seek);
      updateButton();
    }));
    video.addEventListener('timeupdate', () => {
      const round = Math.min(Math.floor(Math.max(0, video.currentTime - 1) / 6), 3);
      const time = Math.min(6, Math.max(0, video.currentTime - 1 - round * 6));
      const stage = time < 1 ? 0 : time < 2.3 ? 1 : time < 3.7 ? 2 : 3;
      stages.forEach((button, index) => {
        button.classList.toggle('is-active', index === stage);
        if (index === stage) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
    });
    video.addEventListener('play', updateButton);
    video.addEventListener('pause', updateButton);
    video.addEventListener('error', () => { video.controls = true; });
    if ('IntersectionObserver' in window) {
      const observer = new IntersectionObserver(([entry]) => {
        visible = entry.isIntersecting;
        sync();
      }, { threshold: 0.12 });
      observer.observe(video);
    } else { visible = true; sync(); }
    reduceMotion.addEventListener('change', () => {
      wantsPlayback = !reduceMotion.matches;
      sync();
    });
    if (reduceMotion.matches) video.pause();
  });
})();
