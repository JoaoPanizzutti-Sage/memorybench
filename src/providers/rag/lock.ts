/**
 * Per-container read/write lock for async code.
 * Multiple readers can hold the lock concurrently.
 * A writer gets exclusive access (no readers, no other writers).
 */
export class ContainerLock {
  private writerPromise = new Map<string, Promise<void>>()
  private writerResolve = new Map<string, () => void>()
  private readerCount = new Map<string, number>()
  private readerDrain = new Map<string, { promise: Promise<void>; resolve: () => void }>()

  async readLock(tag: string): Promise<void> {
    while (this.writerPromise.has(tag)) {
      await this.writerPromise.get(tag)
    }
    this.readerCount.set(tag, (this.readerCount.get(tag) || 0) + 1)
  }

  readUnlock(tag: string): void {
    const count = (this.readerCount.get(tag) || 1) - 1
    if (count <= 0) {
      this.readerCount.delete(tag)
      const drain = this.readerDrain.get(tag)
      if (drain) {
        this.readerDrain.delete(tag)
        drain.resolve()
      }
    } else {
      this.readerCount.set(tag, count)
    }
  }

  async writeLock(tag: string): Promise<void> {
    while (this.writerPromise.has(tag)) {
      await this.writerPromise.get(tag)
    }

    let resolve!: () => void
    const promise = new Promise<void>((r) => { resolve = r })
    this.writerPromise.set(tag, promise)
    this.writerResolve.set(tag, resolve)

    const readers = this.readerCount.get(tag) || 0
    if (readers > 0) {
      let drainResolve!: () => void
      const drainPromise = new Promise<void>((r) => { drainResolve = r })
      this.readerDrain.set(tag, { promise: drainPromise, resolve: drainResolve })
      await drainPromise
    }
  }

  writeUnlock(tag: string): void {
    const resolve = this.writerResolve.get(tag)
    this.writerPromise.delete(tag)
    this.writerResolve.delete(tag)
    if (resolve) resolve()
  }
}
