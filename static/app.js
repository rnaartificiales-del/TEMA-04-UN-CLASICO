      const badge = document.getElementById('badge');
        const badgeText = document.getElementById('badge-text');
        const overlay = document.getElementById('overlay');
        const totalEl = document.getElementById('total');
        const fpsEl = document.getElementById('fps');
        const listEl = document.getElementById('objects');

        function setBadge(state, text) {
            badge.className = 'badge ' + state;
            badgeText.textContent = text;
        }

        async function refresh() {
            try {
                const res = await fetch('/stats', { cache: 'no-store' });
                const data = await res.json();

                if (data.status === 'activo') {
                    setBadge('live', 'En vivo');
                    overlay.style.display = 'none';
                } else if (data.status === 'error') {
                    setBadge('error', 'Error');
                    overlay.style.display = 'flex';
                    overlay.textContent = data.message || 'Error en la fuente de video';
                } else {
                    setBadge('wait', 'Iniciando');
                    overlay.style.display = 'flex';
                    overlay.textContent = 'Cargando modelo...';
                }

                totalEl.textContent = data.total;
                fpsEl.textContent = data.fps;

                const entries = Object.entries(data.objects);
                listEl.innerHTML = '';
                if (entries.length === 0) {
                    listEl.innerHTML = '<li class="empty">Sin detecciones</li>';
                } else {
                    entries.sort((a, b) => b[1] - a[1]).forEach(([name, n]) => {
                        const li = document.createElement('li');
                        const label = document.createElement('span');
                        label.textContent = name;
                        const count = document.createElement('strong');
                        count.textContent = n;
                        li.appendChild(label);
                        li.appendChild(count);
                        listEl.appendChild(li);
                    });
                }
            } catch (e) {
                setBadge('error', 'Sin conexión');
            }
        }

        refresh();
        setInterval(refresh, 1000);