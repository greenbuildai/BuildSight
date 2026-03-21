import { useEffect, useRef, useState } from 'react';

const boundaryHeader = '\r\n\r\n';
const crlf = '\r\n';

const textEncoder = new TextEncoder();
const boundaryHeaderBytes = textEncoder.encode(boundaryHeader);
const crlfBytes = textEncoder.encode(crlf);

const supportsWebGL = () => {
    try {
        const canvas = document.createElement('canvas');
        return Boolean(
            canvas.getContext('webgl2', { powerPreference: 'high-performance' }) ||
            canvas.getContext('webgl', { powerPreference: 'high-performance' })
        );
    } catch {
        return false;
    }
};

const concatBytes = (a, b) => {
    const out = new Uint8Array(a.length + b.length);
    out.set(a);
    out.set(b, a.length);
    return out;
};

const indexOfSubarray = (haystack, needle, fromIndex = 0) => {
    if (needle.length === 0) {
        return -1;
    }
    for (let i = fromIndex; i <= haystack.length - needle.length; i += 1) {
        let matched = true;
        for (let j = 0; j < needle.length; j += 1) {
            if (haystack[i + j] !== needle[j]) {
                matched = false;
                break;
            }
        }
        if (matched) {
            return i;
        }
    }
    return -1;
};

const createShader = (gl, type, source) => {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        const error = gl.getShaderInfoLog(shader);
        gl.deleteShader(shader);
        throw new Error(error || 'Shader compilation failed');
    }
    return shader;
};

const createProgram = (gl, vertexSource, fragmentSource) => {
    const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const error = gl.getProgramInfoLog(program);
        gl.deleteProgram(program);
        throw new Error(error || 'Program link failed');
    }
    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);
    return program;
};

const resizeCanvasToDisplaySize = (canvas, dpr) => {
    const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
    }
    return { width, height };
};

const getBoundary = (contentType) => {
    if (!contentType) {
        return 'frame';
    }
    const match = contentType.match(/boundary=([^;]+)/i);
    if (!match) {
        return 'frame';
    }
    return match[1].trim();
};

const decodeJpeg = async (bytes) => {
    if (typeof ImageDecoder !== 'undefined') {
        const decoder = new ImageDecoder({ data: bytes, type: 'image/jpeg' });
        const result = await decoder.decode();
        decoder.close();
        return result.image;
    }
    return createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
};

export default function GpuVideoFeed({ src, className, alt = 'Processed video stream' }) {
    const canvasRef = useRef(null);
    const glRef = useRef(null);
    const programRef = useRef(null);
    const textureRef = useRef(null);
    const scaleLocationRef = useRef(null);
    const abortRef = useRef(null);
    const [useFallback, setUseFallback] = useState(false);

    useEffect(() => {
        setUseFallback(false);
    }, [src]);

    useEffect(() => {
        if (useFallback && abortRef.current) {
            abortRef.current.abort();
        }
    }, [useFallback]);

    useEffect(() => {
        if (!src) {
            return undefined;
        }

        if (!supportsWebGL()) {
            setUseFallback(true);
            return undefined;
        }

        const canvas = canvasRef.current;
        if (!canvas) {
            return undefined;
        }

        const gl =
            canvas.getContext('webgl2', { alpha: false, antialias: false, powerPreference: 'high-performance' }) ||
            canvas.getContext('webgl', { alpha: false, antialias: false, powerPreference: 'high-performance' });

        if (!gl) {
            setUseFallback(true);
            return undefined;
        }

        glRef.current = gl;

        const vertexSource = `
            attribute vec2 a_position;
            attribute vec2 a_texCoord;
            uniform vec2 u_scale;
            varying vec2 v_texCoord;
            void main() {
                vec2 scaled = a_position * u_scale;
                gl_Position = vec4(scaled, 0.0, 1.0);
                v_texCoord = a_texCoord;
            }
        `;

        const fragmentSource = `
            precision mediump float;
            varying vec2 v_texCoord;
            uniform sampler2D u_texture;
            void main() {
                gl_FragColor = texture2D(u_texture, v_texCoord);
            }
        `;

        try {
            const program = createProgram(gl, vertexSource, fragmentSource);
            programRef.current = program;

            gl.useProgram(program);

            const positionLocation = gl.getAttribLocation(program, 'a_position');
            const texCoordLocation = gl.getAttribLocation(program, 'a_texCoord');
            const scaleLocation = gl.getUniformLocation(program, 'u_scale');
            scaleLocationRef.current = scaleLocation;

            const buffer = gl.createBuffer();
            gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
            const vertices = new Float32Array([
                -1, -1, 0, 1,
                1, -1, 1, 1,
                -1, 1, 0, 0,
                1, 1, 1, 0
            ]);
            gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);

            gl.enableVertexAttribArray(positionLocation);
            gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);

            gl.enableVertexAttribArray(texCoordLocation);
            gl.vertexAttribPointer(texCoordLocation, 2, gl.FLOAT, false, 16, 8);

            const texture = gl.createTexture();
            textureRef.current = texture;
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
            gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
            gl.disable(gl.DEPTH_TEST);
        } catch (error) {
            console.error('[GpuVideoFeed] WebGL init failed', error);
            setUseFallback(true);
            return undefined;
        }

        const controller = new AbortController();
        abortRef.current = controller;

        let decodeInFlight = false;
        let pendingBytes = null;

        const renderFrame = (bitmap) => {
            const glContext = glRef.current;
            const program = programRef.current;
            const texture = textureRef.current;
            const scaleLocation = scaleLocationRef.current;
            const activeCanvas = canvasRef.current;

            if (!glContext || !program || !texture || !scaleLocation || !activeCanvas) {
                if (bitmap.close) {
                    bitmap.close();
                }
                return;
            }

            const { width, height } = resizeCanvasToDisplaySize(activeCanvas, window.devicePixelRatio || 1);
            glContext.viewport(0, 0, width, height);

            const viewAspect = width / height;
            const frameAspect = bitmap.width / bitmap.height;
            let scaleX = 1;
            let scaleY = 1;
            if (viewAspect > frameAspect) {
                scaleX = frameAspect / viewAspect;
            } else {
                scaleY = viewAspect / frameAspect;
            }

            glContext.useProgram(program);
            glContext.uniform2f(scaleLocation, scaleX, scaleY);
            glContext.bindTexture(glContext.TEXTURE_2D, texture);
            glContext.texImage2D(glContext.TEXTURE_2D, 0, glContext.RGBA, glContext.RGBA, glContext.UNSIGNED_BYTE, bitmap);
            glContext.drawArrays(glContext.TRIANGLE_STRIP, 0, 4);

            if (bitmap.close) {
                bitmap.close();
            }
        };

        const decodeAndRender = async (bytes) => {
            if (decodeInFlight) {
                pendingBytes = bytes;
                return;
            }
            decodeInFlight = true;
            try {
                const bitmap = await decodeJpeg(bytes);
                if (!controller.signal.aborted) {
                    renderFrame(bitmap);
                } else if (bitmap.close) {
                    bitmap.close();
                }
            } catch (error) {
                if (!controller.signal.aborted) {
                    console.error('[GpuVideoFeed] Frame decode failed', error);
                }
            } finally {
                decodeInFlight = false;
                if (pendingBytes && !controller.signal.aborted) {
                    const next = pendingBytes;
                    pendingBytes = null;
                    decodeAndRender(next);
                }
            }
        };

        const handlePart = (part) => {
            const headerEnd = indexOfSubarray(part, boundaryHeaderBytes);
            if (headerEnd === -1) {
                return;
            }
            let imageBytes = part.slice(headerEnd + boundaryHeaderBytes.length);
            if (imageBytes.length >= 2 &&
                imageBytes[imageBytes.length - 2] === crlfBytes[0] &&
                imageBytes[imageBytes.length - 1] === crlfBytes[1]) {
                imageBytes = imageBytes.slice(0, -2);
            }
            if (imageBytes.length > 0) {
                decodeAndRender(imageBytes);
            }
        };

        const startStream = async () => {
            try {
                const response = await fetch(src, {
                    signal: controller.signal,
                    cache: 'no-store'
                });
                if (!response.ok || !response.body) {
                    throw new Error(`Stream failed: ${response.status}`);
                }
                const boundary = `--${getBoundary(response.headers.get('content-type'))}`;
                const boundaryBytes = textEncoder.encode(boundary);

                const reader = response.body.getReader();
                let buffer = new Uint8Array(0);

                while (!controller.signal.aborted) {
                    const { value, done } = await reader.read();
                    if (done) {
                        break;
                    }
                    if (value) {
                        buffer = concatBytes(buffer, value);
                        let boundaryIndex = indexOfSubarray(buffer, boundaryBytes);
                        while (boundaryIndex !== -1) {
                            const part = buffer.slice(0, boundaryIndex);
                            if (part.length) {
                                handlePart(part);
                            }
                            buffer = buffer.slice(boundaryIndex + boundaryBytes.length);
                            if (buffer.length >= 2 &&
                                buffer[0] === crlfBytes[0] &&
                                buffer[1] === crlfBytes[1]) {
                                buffer = buffer.slice(2);
                            }
                            boundaryIndex = indexOfSubarray(buffer, boundaryBytes);
                        }
                    }
                }
            } catch (error) {
                if (!controller.signal.aborted) {
                    console.error('[GpuVideoFeed] Stream failed', error);
                    setUseFallback(true);
                }
            }
        };

        startStream();

        return () => {
            controller.abort();
        };
    }, [src]);

    if (useFallback) {
        return <img src={src} alt={alt} className={className} style={{ willChange: 'transform', transform: 'translateZ(0)' }} />;
    }

    return <canvas ref={canvasRef} className={className} aria-label={alt} style={{ willChange: 'transform', transform: 'translateZ(0)' }} />;
}
