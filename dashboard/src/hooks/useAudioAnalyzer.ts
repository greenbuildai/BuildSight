import { useEffect, useRef, useState } from 'react'

export interface AudioMetrics {
  amplitude: number
  frequencyData: Uint8Array
}

export const useAudioAnalyzer = (audioElement: HTMLAudioElement | null, micStream: MediaStream | null) => {
  const [metrics, setMetrics] = useState<AudioMetrics>({
    amplitude: 0,
    frequencyData: new Uint8Array(0),
  })

  const contextRef = useRef<AudioContext | null>(null)
  const analyzerRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaElementAudioSourceNode | MediaStreamAudioSourceNode | null>(null)
  const animationRef = useRef<number>(0)

  useEffect(() => {
    if (!audioElement && !micStream) {
      setMetrics({ amplitude: 0, frequencyData: new Uint8Array(0) })
      return
    }

    const initAnalyzer = () => {
      if (!contextRef.current) {
        contextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)()
      }

      const context = contextRef.current
      if (!analyzerRef.current) {
        analyzerRef.current = context.createAnalyser()
        analyzerRef.current.fftSize = 256
      }

      const analyzer = analyzerRef.current
      const bufferLength = analyzer.frequencyBinCount
      const dataArray = new Uint8Array(bufferLength)

      // Connect source
      if (micStream) {
        if (sourceRef.current) sourceRef.current.disconnect()
        sourceRef.current = context.createMediaStreamSource(micStream)
        sourceRef.current.connect(analyzer)
      } else if (audioElement) {
        if (sourceRef.current) sourceRef.current.disconnect()
        try {
          sourceRef.current = context.createMediaElementSource(audioElement)
          sourceRef.current.connect(analyzer)
          analyzer.connect(context.destination)
        } catch (e) {
          console.warn('[useAudioAnalyzer] Source already connected or failed:', e)
        }
      }

      const update = () => {
        analyzer.getByteFrequencyData(dataArray)
        
        // Calculate average amplitude (0 to 1)
        let sum = 0
        for (let i = 0; i < bufferLength; i++) {
          sum += dataArray[i]
        }
        const avg = sum / bufferLength / 255

        setMetrics({
          amplitude: avg,
          frequencyData: new Uint8Array(dataArray),
        })

        animationRef.current = requestAnimationFrame(update)
      }

      if (context.state === 'suspended') {
        context.resume().then(() => {
          animationRef.current = requestAnimationFrame(update)
        })
      } else {
        animationRef.current = requestAnimationFrame(update)
      }
    }

    initAnalyzer()

    return () => {
      cancelAnimationFrame(animationRef.current)
      if (sourceRef.current) {
        sourceRef.current.disconnect()
      }
    }
  }, [audioElement, micStream])

  return metrics
}
