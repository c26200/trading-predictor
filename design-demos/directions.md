# TradeScope AI — Rediseño 2026

## Spec (común a las tres direcciones)

**Qué es.** Dashboard de un sistema de predicción bursátil: ensemble PyTorch (LSTM
bidireccional con atención + transformer temporal), servido por FastAPI y transmitido por
WebSocket cada 5 segundos. Corre en modo simulación cuando no hay modelos entrenados.

**Audiencia.** Reclutadores técnicos y jurados de competición que abren el proyecto durante
dos minutos, y alguien que quiera entender cómo "piensa" el sistema.

**Mensaje central.** Un sistema que se puede ver pensando: precio, señal, confianza y riesgo
en el mismo campo de visión.

**Tono.** Preciso y alerta, no casino. Autoridad técnica, cero adrenalina de trading.

**Formato.** Una sola pantalla de dashboard a 1440×900, funcional en escritorio.

**Contenido real (no lorem).** Instrumento AAPL a $185.12; predicción $182.28 con dirección
DOWN; señal HOLD; RSI 53.93; MACD 0.09 / 0.10 / 0.16; BB width 0.06; portafolio $100,000
con retorno 0.00 %, Sharpe / Sortino / Calmar 0.000, win rate 0.0 %, 0 operaciones; curva de
equity entre 98.000 y 99.000; 6 instrumentos; log de operaciones vacío ("Awaiting signals");
modo SIMULATION.

**Restricciones.** Sin glow, sin gradientes morados, sin iconos emoji, sin tarjetas con
sombra flotante. Números en cifras tabulares. Contraste de cuerpo ≥ 4.5:1. Nada de datos
inventados: el modo simulación se declara en pantalla.

---

## Candidatos

### 1 · Best designer — Studio Dumbar
- **Lógica y fuente.** Estudio holandés (Dumbar, Ámsterdam) conocido por tipografía a escala
  monumental, color como señal y composición asimétrica.
- **Qué se toma.** Contraste de escala brutal (precio a 160 px), un solo acento eléctrico
  contra tinta profunda, reglas de 1 px en lugar de tarjetas, rail vertical de instrumentos.
- **Material.** Tipografía + series de precio generadas en SVG.
- **Esqueleto.** Grid asimétrico: rail 200 px · campo de mercado 1fr · panel de señal 320 px.
- **Dónde se gasta la audacia.** El precio ocupa el campo como un titular de periódico; la
  señal HOLD es un bloque sólido de color con tipografía negra.
- **Form.** El precio es el protagonista de una pantalla de trading; todo lo demás cede.

### 2 · Best designer — Irma Boom
- **Lógica y fuente.** Diseñadora de libros holandesa; densidad tipográfica, pliegos, hairlines,
  ritmo de lectura impreso.
- **Qué se toma.** Retícula de boletín, etiquetas en versalitas, filetes finos, una tinta de
  acento, jerarquía por cuerpo y no por tarjetas.
- **Material.** Datos como tabla editorial + un grabado de mercado (línea) como lámina.
- **Esqueleto.** Cabecera de boletín, tres columnas densas, lámina de chart a sangre.
- **Dónde se gasta la audacia.** La pantalla se lee como un informe de mercado impreso.
- **Form.** El producto es un informe de predicción; leerlo como informe hace legible la señal.

### 3 · Referencia real — Fey (fey.com)
- **Lógica y fuente.** Interfaz financiera de Fey: densidad alta, jerarquía callada, cifras
  tabulares, filetes mínimos. Por verificar con captura.
- **Qué se toma.** Panel de datos sin ruido, etiquetas pequeñas, alineación decimal, un solo
  módulo que cambia de estado.
- **Material.** Tablas y charts SVG.
- **Esqueleto.** Rail persistente, cabecera de instrumento, módulos con hairlines.
- **Dónde se gasta la audacia.** Módulo de señal a sangre completa con estado cromático.
- **Form.** Un profesional escanea números; la alineación decimal es la función.

### 4 · Referencia real — Mercury (mercury.com)
- **Lógica y fuente.** Fintech de banca para startups: papel frío, aire, serif de display,
  confianza sin adornos. Por verificar con captura.
- **Qué se toma.** Fondo papel, mucho aire, serif de alto contraste solo en el precio,
  acento verde profundo, tarjetas con borde de 1 px y sin sombra.
- **Material.** Tipografía + área de chart con lavado tenue.
- **Esqueleto.** Nav superior, banda de precio, tres módulos, pie con equity y log.
- **Dónde se gasta la audacia.** Precio en serif a 96 px; el resto en silencio.
- **Form.** Monitoreo de sesiones largas: calma, no urgencia.

### 5 · Ruleta — web #5 Pure-CSS Art (Lynn Fisher)
- **Lógica y fuente.** Lynn Fisher (lynnandtonic.com), ilustración geométrica en CSS puro,
  cero imágenes, formas que se reordenan.
- **Qué se toma.** Planos de color plano, geometría dibujada con divs/SVG, paleta de 3-4
  colores, cero iconos.
- **Material.** Geometría programática: velas como rectángulos, retícula como arquitectura.
- **Esqueleto.** Composición geométrica modular con reglas gruesas.
- **Dónde se gasta la audacia.** El chart entero es una composición plana sin gradientes.
- **Form.** El mercado es geometría; dibujarlo como geometría es honesto.

### 6 · Ruleta — east #5 Experimental Hanzi Poster
- **Lógica y fuente.** Cartel tipográfico experimental chino: escala extrema, glifos
  deconstruidos, energía de póster.
- **Qué se toma.** El símbolo del instrumento como objeto monumental; datos en columnas
  pequeñas en los bordes; paleta limitada.
- **Material.** Tipografía como imagen.
- **Esqueleto.** "AAPL" a 300 px como composición; métricas en los márgenes.
- **Dónde se gasta la audacia.** El ticker como escultura tipográfica.
- **Form.** Un solo instrumento es el sujeto; ampliarlo lo convierte en el tema.

---

## Selección para versión completa

1. **Dirección 1 — Signal Field** (Dumbar). Familia: tinta profunda + acento eléctrico, sans display.
2. **Dirección 2 — Market Bulletin** (Mercury). Familia: papel frío + serif display.
3. **Dirección 3 — Market Geometry** (Pure-CSS). Familia: planos de color plano, geometría, sans geométrica.

Descartadas para completa (quedan como miniaturas): Irma Boom, Fey, Hanzi Poster.
