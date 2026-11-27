import { useCallback, useEffect, useState } from 'react';
import Icon from '../components/Icon';
import PoseCanvas from '../components/PoseCanvas';
import { Panel, PageHeading, MetaStrip, Pill, Eyebrow } from '../components/ui';
import { timeLabel } from '../utils';
import { getCameraStatus, setCameraRunning } from '../services/api';

export default function LiveMonitoring({ vm, th, updatedAt, accessToken, onLogout }) {
  const { pose } = vm;
  const people = vm.people || [];
  const performance = vm.performance || {};
  const [camera, setCamera] = useState({ running: false, frames_processed: 0, last_error: null });
  const [cameraBusy, setCameraBusy] = useState(false);
  const [cameraError, setCameraError] = useState('');
  const score = pose?.score;
  const fall = pose?.fallConfirmed, high = !fall && pose?.level === 'High';

  const refreshCamera = useCallback(async () => {
    try {
      setCamera(await getCameraStatus(accessToken));
    } catch (error) {
      if (error.status === 401) onLogout();
      else setCameraError(error.message);
    }
  }, [accessToken, onLogout]);

  useEffect(() => {
    refreshCamera();
    const timer = setInterval(refreshCamera, 2000);
    return () => clearInterval(timer);
  }, [refreshCamera]);

  const toggleCamera = async () => {
    setCameraBusy(true);
    setCameraError('');
    try {
      await setCameraRunning(accessToken, !camera.running);
      await refreshCamera();
    } catch (error) {
      if (error.status === 401) onLogout();
      else setCameraError(error.message);
    } finally {
      setCameraBusy(false);
    }
  };

  return (
    <>
      <PageHeading title="Live Monitoring" sub="Real-time pose tracking and fall-risk monitoring"
        right={<button className={`btn ${camera.running ? 'danger' : 'primary'}`} onClick={toggleCamera} disabled={cameraBusy}>
          <Icon name={camera.running ? 'x' : 'camera'} size={16} />
          {cameraBusy ? 'Please wait…' : camera.running ? 'Stop camera' : 'Start camera'}
        </button>} />
      {cameraError && <div className="conn-banner" role="alert">{cameraError}</div>}
      {camera.last_error && <div className="conn-banner" role="alert">Camera stopped: {camera.last_error}</div>}
      <div className="focus-head">
        <div><Eyebrow>{camera.running ? 'CAMERA RUNNING' : 'FOCUSED VIEW'}</Eyebrow><h2 className="h-md">{pose ? `${pose.room || '—'} · ${pose.patient || '—'}` : camera.running ? 'Waiting for first frame…' : 'No active feed'}</h2></div>
        {score != null && <Pill level={pose.level}>{score}%</Pill>}
      </div>
      {fall && (
        <div className="banner fall-banner"><Icon name="alert" size={18} />
          <div><b>Fall detected</b><span>The model reports a confirmed fall for {pose.patient || 'the tracked patient'}. Check the patient immediately.</span></div></div>
      )}
      {high && (
        <div className="banner warn-banner"><Icon name="bell" size={18} />
          <div><b>High risk state detected</b><span>Monitor the patient closely. Risk engine score exceeds the configured {th.high.toFixed(2)} threshold.</span></div></div>
      )}
      {people.length > 0 ? (
        <>
          <MetaStrip items={[
            [performance.tracked_people ?? people.filter(p => p.trackingStatus === 'TRACKED').length, 'Tracked people'],
            [pose?.fps || '—', 'FPS'],
            [performance.yolo_inference_ms == null ? '—' : `${performance.yolo_inference_ms} ms`, 'YOLO latency'],
            [performance.pipeline_latency_ms == null ? '—' : `${performance.pipeline_latency_ms} ms`, 'Pipeline latency'],
          ]} />
          <div className="person-grid">
            {people.map(person => {
              const isLost = person.trackingStatus === 'LOST';
              const personName = `P${person.personId}`;
              return (
                <Panel key={person.personId} className="person-card">
                  <div className="person-card-head">
                    <div><Eyebrow>{personName} · {isLost ? 'TRACKING LOST' : 'TRACKED'}</Eyebrow>
                      <h3 className="h-md">{person.prediction || 'Awaiting prediction'}</h3></div>
                    <Pill level={isLost ? 'Away' : person.level}>
                      {person.score == null ? '—' : `${person.score}%`}
                    </Pill>
                  </div>
                  <PoseCanvas
                    keypoints={person.keypoints}
                    height={240}
                    tag={`${personName} · ${person.prediction || person.trackingStatus}`}
                    emptyText={isLost ? `Last seen ${person.framesSinceSeen} frames ago` : 'Waiting for pose keypoints…'}
                  />
                  <div className="person-meta">
                    <div><small>Status</small><b>{isLost ? 'Tracking lost' : person.fallConfirmed ? 'FALL DETECTED' : person.prediction || 'NORMAL'}</b></div>
                    <div><small>Risk</small><b>{person.score == null ? '—' : `${person.score}%`}</b></div>
                    <div><small>Confidence</small><b>{person.detConf == null ? '—' : `${person.detConf}%`}</b></div>
                    <div><small>Frames missing</small><b>{person.framesSinceSeen}</b></div>
                  </div>
                </Panel>
              );
            })}
          </div>
        </>
      ) : (
        <div className="live-grid">
          <Panel className="flush">
            <PoseCanvas keypoints={pose?.keypoints} height={340} aspect={pose?.aspect || 16 / 9} tag={pose?.keypoints ? `${(pose.patient || 'PERSON').toUpperCase()}${pose.detConf != null ? ` · ${pose.detConf}%` : ''}` : null} />
            <MetaStrip items={[[pose?.fps || '—', 'FPS'], [pose?.frame !== '' && pose?.frame != null ? pose.frame : '—', 'Frame'], [pose?.room || '—', 'Room'], [timeLabel(updatedAt), 'Timestamp']]} />
          </Panel>
          <div className="stack">
            <Panel><Eyebrow>AI PREDICTION</Eyebrow><h3 className="big">{pose?.prediction || '—'}</h3>
              <div className="kv"><span>Confidence</span><b>{pose?.predConf == null ? '—' : `${pose.predConf}%`}</b></div></Panel>
            <Panel><Eyebrow>RISK SCORE</Eyebrow><h3 className={`big ${pose?.level === 'High' ? 'warn' : ''}`}>{score == null ? '—' : `${score}%`}</h3>
              <div className="kv"><span>Risk level</span>{pose?.level ? <Pill level={pose.level} /> : <b>—</b>}</div></Panel>
          </div>
        </div>
      )}
    </>
  );
}
