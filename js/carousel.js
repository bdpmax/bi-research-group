/**
 * Lightweight image carousel — reusable across research areas.
 * Usage: new Carousel(containerElement, { interval: 5000 })
 */
class Carousel {
  constructor(el, opts = {}) {
    this.el = el;
    this.track = el.querySelector('.carousel-track');
    this.slides = Array.from(el.querySelectorAll('.carousel-slide'));
    this.dots = Array.from(el.querySelectorAll('.carousel-dot'));
    this.total = this.slides.length;
    this.index = Math.floor(Math.random() * this.total);
    this.interval = opts.interval || 7000;
    this.timer = null;

    // Arrow buttons
    const prev = el.querySelector('.carousel-btn--prev');
    const next = el.querySelector('.carousel-btn--next');
    if (prev) prev.addEventListener('click', () => this.go(this.index - 1));
    if (next) next.addEventListener('click', () => this.go(this.index + 1));

    // Dot buttons
    this.dots.forEach((dot, i) => {
      dot.addEventListener('click', () => this.go(i));
    });

    // Pause on hover
    el.addEventListener('mouseenter', () => this.pause());
    el.addEventListener('mouseleave', () => this.play());

    // Jump to random start instantly (no transition)
    this.track.style.transition = 'none';
    this.go(this.index);
    // Force reflow, then restore transition
    this.track.offsetHeight;
    this.track.style.transition = '';

    this.play();
  }

  go(i) {
    this.index = ((i % this.total) + this.total) % this.total;
    this.track.style.transform = `translateX(-${this.index * 100}%)`;
    this.dots.forEach((d, j) => d.classList.toggle('active', j === this.index));
  }

  play() {
    this.pause();
    this.timer = setInterval(() => this.go(this.index + 1), this.interval);
  }

  pause() {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }
}

// Auto-init all carousels on the page
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.carousel').forEach(el => new Carousel(el));
});
